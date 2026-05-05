from __future__ import annotations

from typing import Any

import numpy as np

from Core.Chunk.DocChunk import DocChunk
from Core.Common.Logger import logger
from Core.Common.Utils import mdhash_id
from Core.Graph.ERGraph import ERGraph
from Core.Graph.TreeGraph import TreeGraph
from Core.Graph.TreeGraphBalanced import TreeGraphBalanced
from Core.Schema.ChunkSchema import TextChunk
from Core.Storage.NameSpace import Namespace


class RAPPOGraph(ERGraph):
    def __init__(self, config, llm, encoder):
        super().__init__(config.graph, llm, encoder)
        self.full_config = config
        self.tree_graph = self._create_tree_graph(config, llm, encoder)
        self._namespace = None
        self._tree_namespace = None
        self._query_chunk_namespace = None
        self._query_doc_chunk = None

    @staticmethod
    def _create_tree_graph(config, llm, encoder):
        tree_type = getattr(config.graph, "rappo_tree_graph_type", "tree_graph")
        if tree_type == "tree_graph_balanced":
            return TreeGraphBalanced(config, llm, encoder)
        if tree_type != "tree_graph":
            logger.warning(
                f"Unsupported RAPPO tree graph type '{tree_type}'. Falling back to tree_graph."
            )
        return TreeGraph(config, llm, encoder)

    @property
    def namespace(self):
        return self._namespace

    @namespace.setter
    def namespace(self, namespace):
        self._namespace = namespace
        self._graph.namespace = namespace
        self._tree_namespace = Namespace(
            namespace.workspace, f"{namespace.namespace}_rappo_tree"
        )
        self.tree_graph.namespace = self._tree_namespace
        self._query_chunk_namespace = Namespace(
            namespace.workspace, f"{namespace.namespace}_rappo_chunks"
        )
        self._query_doc_chunk = None

    def _collect_leaf_descendants(self, node_index: int, cache: dict[int, tuple[int, ...]]):
        if node_index in cache:
            return cache[node_index]

        node = self.tree_graph._graph.tree.all_nodes[node_index]
        if not node.children:
            cache[node_index] = (node_index,)
            return cache[node_index]

        leaf_indices = []
        for child_index in sorted(node.children):
            leaf_indices.extend(self._collect_leaf_descendants(child_index, cache))

        cache[node_index] = tuple(dict.fromkeys(leaf_indices))
        return cache[node_index]

    def _select_representative_leaves(self, leaf_indices: tuple[int, ...], limit: int):
        leaf_nodes = [self.tree_graph._graph.tree.all_nodes[idx] for idx in leaf_indices]
        if limit <= 0 or len(leaf_nodes) <= limit:
            return leaf_nodes

        embeddings = np.array([node.embedding for node in leaf_nodes], dtype=float)
        if embeddings.size == 0:
            return leaf_nodes[:limit]

        centroid = embeddings.mean(axis=0, keepdims=True)
        metric = getattr(self.full_config.graph, "cluster_metric", "cosine")

        if metric == "cosine":
            embedding_norms = np.linalg.norm(embeddings, axis=1)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm == 0:
                distances = np.linalg.norm(embeddings - centroid, axis=1)
            else:
                safe_norms = np.where(embedding_norms == 0, 1.0, embedding_norms)
                similarities = (embeddings @ centroid.T).reshape(-1) / (
                    safe_norms * centroid_norm
                )
                distances = 1 - similarities
        else:
            distances = np.linalg.norm(embeddings - centroid, axis=1)

        selected_indices = np.argsort(distances, kind="mergesort")[:limit]
        return [leaf_nodes[idx] for idx in selected_indices]

    def _build_text_chunk(self, chunk_id: str, content: str, index: int, title: str):
        clean_content = content.strip()
        return TextChunk(
            tokens=len(self.ENCODER.encode(clean_content)),
            chunk_id=chunk_id,
            content=clean_content,
            doc_id=mdhash_id(f"{title}:{clean_content}", prefix="doc-"),
            index=index,
            title=title,
        )

    async def _build_rappo_chunks(self):
        if self._tree_namespace is None:
            raise RuntimeError("RAPPO tree namespace has not been initialized.")

        tree_storage = self.tree_graph._graph
        leaf_cache = {}
        representative_limit = max(
            0,
            int(
                getattr(
                    self.full_config.graph,
                    "rappo_representative_chunks_per_cluster",
                    1,
                )
            ),
        )

        summary_specs = []
        representative_specs = []
        seen_representative_ids = set()

        for layer in range(1, tree_storage.num_layers):
            for node in tree_storage.get_layer(layer):
                if not node.children:
                    continue

                summary_text = node.text.strip()
                if summary_text:
                    summary_specs.append(
                        {
                            "chunk_id": mdhash_id(
                                f"summary:{node.index}:{summary_text}",
                                prefix="rappo-summary-",
                            ),
                            "content": summary_text,
                            "title": f"RAPPO summary layer {layer} node {node.index}",
                        }
                    )

                if representative_limit <= 0:
                    continue

                leaf_indices = self._collect_leaf_descendants(node.index, leaf_cache)
                for leaf in self._select_representative_leaves(
                    leaf_indices, representative_limit
                ):
                    representative_text = leaf.text.strip()
                    if not representative_text:
                        continue
                    representative_id = mdhash_id(
                        f"rep:{leaf.index}:{representative_text}", prefix="rappo-rep-"
                    )
                    if representative_id in seen_representative_ids:
                        continue
                    seen_representative_ids.add(representative_id)
                    representative_specs.append(
                        {
                            "chunk_id": representative_id,
                            "content": representative_text,
                            "title": f"RAPPO representative chunk {leaf.index}",
                        }
                    )

        if not summary_specs and not representative_specs:
            logger.warning(
                "RAPPO did not find parent nodes in the RAPTOR tree. Falling back to leaf chunks only."
            )
            for leaf in tree_storage.leaf_nodes:
                representative_text = leaf.text.strip()
                if not representative_text:
                    continue
                representative_specs.append(
                    {
                        "chunk_id": mdhash_id(
                            f"rep:{leaf.index}:{representative_text}",
                            prefix="rappo-rep-",
                        ),
                        "content": representative_text,
                        "title": f"RAPPO representative chunk {leaf.index}",
                    }
                )

        chunk_specs = summary_specs + representative_specs
        chunk_pairs = []
        for index, spec in enumerate(chunk_specs):
            chunk_pairs.append(
                (
                    spec["chunk_id"],
                    self._build_text_chunk(
                        chunk_id=spec["chunk_id"],
                        content=spec["content"],
                        index=index,
                        title=spec["title"],
                    ),
                )
            )

        logger.info(
            "RAPPO prepared {summary_count} summary chunks and {rep_count} representative chunks "
            "({total_count} total).".format(
                summary_count=len(summary_specs),
                rep_count=len(representative_specs),
                total_count=len(chunk_pairs),
            )
        )
        return chunk_pairs

    async def _persist_query_doc_chunk(self, chunk_pairs):
        query_doc_chunk = DocChunk(
            self.full_config.chunk, self.ENCODER, self._query_chunk_namespace
        )
        query_doc_chunk._chunk._data.clear()
        query_doc_chunk._chunk._chunk.clear()
        query_doc_chunk._chunk._key_to_index.clear()

        for chunk_id, chunk in chunk_pairs:
            await query_doc_chunk._chunk.upsert(chunk_id, chunk)

        await query_doc_chunk._chunk.persist()
        self._query_doc_chunk = query_doc_chunk
        return query_doc_chunk

    async def _build_graph(self, chunk_list: list[Any]):
        logger.info("RAPPO: building the internal RAPTOR tree graph.")
        await self.tree_graph.build_graph(chunk_list, self.full_config.graph.force)

        logger.info("RAPPO: deriving synthetic chunks from RAPTOR summaries and representatives.")
        rappo_chunks = await self._build_rappo_chunks()
        await self._persist_query_doc_chunk(rappo_chunks)

        logger.info("RAPPO: building the derived ER graph from synthetic RAPTOR chunks.")
        await super()._build_graph(rappo_chunks)

    async def ensure_tree_graph_loaded(self, source_doc_chunk=None):
        if self._tree_namespace is None:
            raise RuntimeError("RAPPO tree namespace has not been initialized.")

        tree_storage = self.tree_graph._graph
        if getattr(tree_storage, "num_nodes", 0) > 0:
            return True

        tree_loaded = await tree_storage.load_tree_graph(False)
        if tree_loaded:
            return True

        if source_doc_chunk is None:
            return False

        logger.info(
            "RAPPO: internal RAPTOR tree not found in memory. Rebuilding it for query-time tree retrieval."
        )
        await self.tree_graph.build_graph(
            await source_doc_chunk.get_chunks(), self.full_config.graph.force
        )
        return True

    async def get_query_doc_chunk(self, source_doc_chunk):
        if self._query_doc_chunk is not None:
            return self._query_doc_chunk

        query_doc_chunk = DocChunk(
            self.full_config.chunk, self.ENCODER, self._query_chunk_namespace
        )
        if await query_doc_chunk._load_chunk(force=False):
            self._query_doc_chunk = query_doc_chunk
            logger.info("RAPPO: loaded the persisted synthetic chunk store.")
            return query_doc_chunk

        logger.info("RAPPO: synthetic chunk store not found. Reconstructing it now.")
        tree_loaded = await self.tree_graph._graph.load_tree_graph(False)
        if not tree_loaded:
            await self.tree_graph.build_graph(
                await source_doc_chunk.get_chunks(), self.full_config.graph.force
            )

        rappo_chunks = await self._build_rappo_chunks()
        return await self._persist_query_doc_chunk(rappo_chunks)
