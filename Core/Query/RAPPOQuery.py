from Core.Common.Constants import Retriever
from Core.Common.Logger import logger
from Core.Common.Memory import Memory
from Core.Prompt import QueryPrompt
from Core.Query.BaseQuery import BaseQuery
from Core.Retriever.EntitiyRetriever import EntityRetriever
from Core.Schema.Message import Message


class RAPPOQuery(BaseQuery):
    def __init__(self, config, retriever_context):
        super().__init__(config, retriever_context)
        self._tree_entity_retriever = None

        tree_graph = retriever_context.as_dict.get("rappo_tree_graph")
        tree_entities_vdb = retriever_context.as_dict.get("rappo_tree_entities_vdb")
        retriever_config = retriever_context.as_dict.get("config")
        if tree_graph is not None and tree_entities_vdb is not None and retriever_config is not None:
            self._tree_entity_retriever = EntityRetriever(
                config=retriever_config,
                graph=tree_graph,
                entities_vdb=tree_entities_vdb,
            )

    async def reason_step(self, few_shot: list, query: str, passages: list, thoughts: list):
        prompt_demo = ""
        for sample in few_shot:
            prompt_demo += (
                f'{sample["document"]}\n\nQuestion: {sample["question"]}\n'
                f'Thought: {sample["answer"]}\n\n'
            )

        prompt_user = ""
        for passage in passages:
            prompt_user += f"{passage}\n\n"
        prompt_user += f"Question: {query} \n Thought:" + " ".join(thoughts)

        try:
            response_content = await self.llm.aask(
                msg=prompt_demo + prompt_user,
                system_msgs=[QueryPrompt.IRCOT_REASON_INSTRUCTION],
            )
        except Exception as e:
            print(e)
            return ""
        return response_content

    async def _retrieve_tree_nodes(self, query):
        tree_top_k = max(0, int(getattr(self.config, "rappo_tree_top_k", 2)))
        if tree_top_k == 0 or self._tree_entity_retriever is None:
            return []

        try:
            tree_nodes = await self._tree_entity_retriever.retrieve_relevant_content(
                seed=query,
                tree_node=True,
                top_k=tree_top_k,
                mode="vdb",
            )
            return tree_nodes or []
        except Exception as e:
            logger.exception(f"Failed to retrieve RAPPO tree nodes: {e}")
            return []

    async def _retrieve_relevant_contexts(self, query):
        entities = await self.extract_query_entities(query)
        if not self.config.augmentation_ppr:
            retrieved_passages, scores = await self._retriever.retrieve_relevant_content(
                query=query,
                seed_entities=entities,
                link_entity=True,
                type=Retriever.CHUNK,
                mode="ppr",
            )
            thoughts = []

            passage_scores = {
                passage: score for passage, score in zip(retrieved_passages, scores)
            }
            few_shot_examples = []
            for iteration in range(2, self.config.max_ir_steps + 1):
                logger.info("Entering the ir-cot iteration: {}".format(iteration))
                new_thought = await self.reason_step(
                    few_shot_examples,
                    query,
                    retrieved_passages[: self.config.top_k],
                    thoughts,
                )
                thoughts.append(new_thought)

                if "So the answer is:" in new_thought:
                    break

                new_passages, new_scores = await self._retriever.retrieve_relevant_content(
                    query=query,
                    seed_entities=thoughts,
                    link_entity=True,
                    type=Retriever.CHUNK,
                    mode="ppr",
                )

                for passage, score in zip(new_passages, new_scores):
                    if passage in passage_scores:
                        passage_scores[passage] = max(passage_scores[passage], score)
                    else:
                        passage_scores[passage] = score

                sorted_passages = sorted(
                    passage_scores.items(), key=lambda item: item[1], reverse=True
                )
                retrieved_passages, scores = zip(*sorted_passages)

            return {
                "passages": list(retrieved_passages),
                "tree_nodes": await self._retrieve_tree_nodes(query),
            }

        return {
            "passages": await self._retriever.retrieve_relevant_content(
                query=query,
                seed_entities=entities,
                type=Retriever.CHUNK,
                mode="aug_ppr",
            ),
            "tree_nodes": await self._retrieve_tree_nodes(query),
        }

    async def generation_qa(self, query, context):
        if context is None:
            return QueryPrompt.FAIL_RESPONSE

        tree_nodes = []
        rappo_context = context
        if isinstance(context, dict):
            tree_nodes = context.get("tree_nodes", []) or []
            rappo_context = context.get("passages")

        if self.config.augmentation_ppr:
            combined_context = rappo_context
            if tree_nodes:
                tree_section = "\n\n".join(tree_nodes)
                combined_context = (
                    "-----RAPTOR Tree Nodes-----\n"
                    f"{tree_section}\n\n"
                    "-----RAPPO Graph Context-----\n"
                    f"{rappo_context}"
                )
            msg = QueryPrompt.GENERATE_RESPONSE_QUERY_WITH_REFERENCE.format(
                query=query, context=combined_context
            )
            return await self.llm.aask(msg=msg)

        retrieved_passages = []
        if rappo_context:
            retrieved_passages = list(rappo_context[: self.config.num_doc])
        working_memory = Memory()
        instruction = (
            QueryPrompt.COT_SYSTEM_DOC
            if len(retrieved_passages) or len(tree_nodes)
            else QueryPrompt.COT_SYSTEM_NO_DOC
        )
        working_memory.add(Message(content=instruction, role="system"))
        user_prompt = ""
        if tree_nodes:
            user_prompt += "RAPTOR Retrieved Tree Nodes:\n"
            for tree_node in tree_nodes:
                user_prompt += f" {tree_node}\n\n"
        if retrieved_passages:
            user_prompt += "RAPPO Retrieved Passages:\n"
        for passage in retrieved_passages:
            user_prompt += f" {passage}\n\n"
        user_prompt += "Question: " + query + "\nThought: "
        working_memory.add(Message(content=user_prompt, role="user"))

        system_msgs = "\n".join(
            f"{msg.sent_from}: {msg.content}" for msg in working_memory.get()
        )
        augmented_prompt = f"system_msgs:\n{system_msgs}\n\nmsg:\n{user_prompt}"
        try:
            response = await self.llm.aask(msg=user_prompt, system_msgs=[system_msgs])
        except Exception as e:
            print("QA read exception", e)
            self._record_prompt_trace(
                method="RAPPO",
                query=query,
                augmented_prompt=augmented_prompt,
                answer="",
            )
            return ""
        self._record_prompt_trace(
            method="RAPPO",
            query=query,
            augmented_prompt=augmented_prompt,
            answer=response,
        )
        return response

    async def generation_summary(self, query, context):
        pass
