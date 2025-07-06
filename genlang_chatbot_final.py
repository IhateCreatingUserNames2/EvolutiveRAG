import os
import pickle
import numpy as np
import logging
import uuid
import json
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

# --- Configurações Iniciais ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GenlangChatbot")


# ... (Classes GenlangVector, CognitiveState, ChatResponseModel, CriticalityGovernor, MemorySystem permanecem idênticas) ...
@dataclass
class GenlangVector:
    vector: np.ndarray
    source_text: str
    source_agent: str
    self_eval_score: float

    def similarity(self, other: 'GenlangVector') -> float:
        norm_v1, norm_v2 = np.linalg.norm(self.vector), np.linalg.norm(other.vector)
        return 0.0 if norm_v1 == 0 or norm_v2 == 0 else float(np.dot(self.vector, other.vector) / (norm_v1 * norm_v2))


class CognitiveState:
    RIGID = "RÍGIDO (repetitivo, precisa de mais criatividade)"
    OPTIMAL = "ÓTIMO (equilibrado e relevante)"
    CHAOTIC = "CAÓTICO (desconexo, precisa de mais foco)"


class ChatResponseModel(BaseModel):
    final_response: str = Field(description="A resposta final e polida para o usuário.")
    self_evaluation_score: float = Field(
        description="Uma nota de 0.0 a 1.0 avaliando a qualidade da própria resposta, considerando relevância, clareza e uso da memória.",
        ge=0.0, le=1.0)
    thought_process: str = Field(description="Um breve resumo do raciocínio para chegar à resposta.")
    extracted_facts: Optional[Dict[str, str]] = Field(
        description="Um dicionário de fatos explícitos extraídos da fala do usuário (ex: {'user_name': 'João', 'user_preference_animal': 'gatos'}). Retorne null se nenhum fato novo for revelado.")


class CriticalityGovernor:
    def __init__(self, history_size: int = 50):
        self.communication_history = deque(maxlen=history_size)
        self.task_success_history = deque(maxlen=history_size)

    def record_turn(self, thought_vector: GenlangVector, self_eval_score: float):
        self.communication_history.append(thought_vector);
        self.task_success_history.append(self_eval_score)

    def assess_and_govern(self, current_temp: float) -> Tuple[float, str, Dict[str, float]]:
        if len(self.communication_history) < 10: return current_temp, CognitiveState.OPTIMAL, {"novelty": 0.5,
                                                                                               "coherence": 0.5,
                                                                                               "grounding": 0.5}
        metrics = {"novelty": self._calculate_novelty(), "coherence": self._calculate_coherence(),
                   "grounding": self._calculate_grounding()}
        new_temp, state = current_temp, CognitiveState.OPTIMAL
        if metrics["novelty"] < 0.35 and metrics["coherence"] > 0.6:
            state, new_temp = CognitiveState.RIGID, min(current_temp * 1.25, 1.3); logger.warning(
                f"Estado RÍGIDO. Temp -> {new_temp:.2f}")
        elif metrics["novelty"] > 0.7 and metrics["coherence"] < 0.4:
            state, new_temp = CognitiveState.CHAOTIC, max(current_temp * 0.75, 0.1); logger.warning(
                f"Estado CAÓTICO. Temp -> {new_temp:.2f}")
        return new_temp, state, metrics

    def _calculate_novelty(self) -> float:
        if len(self.communication_history) < 10: return 0.5
        recent, older = list(self.communication_history)[-5:], list(self.communication_history)[:-5];
        return np.mean([1 - r.similarity(o) for r in recent for o in older]) if older else 0.8

    def _calculate_coherence(self) -> float:
        if len(self.communication_history) < 2: return 0.5
        recent = list(self.communication_history)[-10:];
        return np.mean([recent[i].similarity(recent[i + 1]) for i in range(len(recent) - 1)]) if len(
            recent) > 1 else 0.5

    def _calculate_grounding(self) -> float:
        return np.mean(list(self.task_success_history)[-20:]) if self.task_success_history else 0.5


class MemorySystem:
    def __init__(self, cluster_threshold: float = 0.85):
        self.concept_clusters: Dict[str, List[GenlangVector]] = {}; self.cluster_threshold = cluster_threshold

    def store_concept(self, vector: GenlangVector):
        if not self.concept_clusters: self.concept_clusters["concept_0"] = [vector]; return
        similarities = {cid: np.mean([vector.similarity(v) for v in c_vectors]) for cid, c_vectors in
                        self.concept_clusters.items()}
        if not similarities: self.concept_clusters[f"concept_{len(self.concept_clusters)}"] = [vector]; return
        best_cluster, best_sim = max(similarities.items(), key=lambda item: item[1])
        if best_sim > self.cluster_threshold:
            self.concept_clusters[best_cluster].append(vector)
        else:
            self.concept_clusters[f"concept_{len(self.concept_clusters)}"] = [vector]

    def get_cluster_centroids(self) -> Dict[str, np.ndarray]:
        return {cid: np.mean([v.vector for v in vectors], axis=0) for cid, vectors in self.concept_clusters.items() if
                vectors}

    def get_vocabulary_size(self) -> int:
        return len(self.concept_clusters)


# ===================================================================
# <<< O CHATBOT COM ARQUITETURA HÍBRIDA FINAL >>>
# ===================================================================

class GenlangChatbot:
    def __init__(self, checkpoint_path: str, vector_dim: int = 256):
        self.checkpoint_path = checkpoint_path
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=vector_dim)

        if os.path.exists(checkpoint_path):
            self.load_state(checkpoint_path)
        else:
            logger.info("Nenhum checkpoint encontrado. Iniciando um novo sistema Genlang.")
            self.memory = MemorySystem()
            self.governor = CriticalityGovernor()
            # <<< MUDANÇA CRÍTICA 1: A memória de fatos agora é um único dicionário. >>>
            self.fact_store: Dict[str, str] = {}
            self.system_state = {"llm_temperature": 0.6}

        self.parser = PydanticOutputParser(pydantic_object=ChatResponseModel)
        self.prompt_template = self._create_prompt_template()
        self.session_id = ""
        self.conversation_history = []

    def _create_prompt_template(self) -> ChatPromptTemplate:
        # O prompt não precisa de grandes mudanças, pois a lógica de injeção de fatos é externa.
        return ChatPromptTemplate.from_template(
            """Você é um assistente de IA avançado com dois tipos de memória.

           **INSTRUÇÕES GERAIS:**
           1.  **USE OS FATOS:** Personalize sua resposta usando os "Fatos Conhecidos Sobre o Usuário". Se a pergunta do usuário estiver relacionada a um fato, mencione-o.
           2.  **EXTRAIA NOVOS FATOS:** Se o usuário revelar uma nova informação pessoal (nome, preferência, objetivo), capture-a no campo 'extracted_facts'. Use chaves descritivas como 'user_name' ou 'user_preference_color'.
           3.  **USE A MEMÓRIA DE PROCESSO:** As "Memórias de Conversas Passadas" são exemplos de como raciocinar. Use-as como inspiração para seu 'thought_process'.

           ---
           **Fatos Conhecidos Sobre o Usuário:**
           {known_facts}
           ---
           **Memórias de Conversas Passadas (Inspiração de Processo):**
           {retrieved_context}
           ---
           **Conversa Atual:**
           {history}
           Usuário: {user_input}
           ---

           **Formato de Saída Obrigatório (JSON):**
           {format_instructions}
           """
        )

    # ... _get_llm_instance e _retrieve_context permanecem os mesmos ...
    def _get_llm_instance(self) -> ChatOpenAI:
        return ChatOpenAI(model="gpt-4o-mini", temperature=self.system_state.get("llm_temperature", 0.6))

    def _retrieve_context(self, search_prompt: str, top_k: int = 1) -> str:
        genlang_centroids = self.memory.get_cluster_centroids()
        if not genlang_centroids: return "Nenhuma memória de processo disponível."
        query_vector = np.array(self.embedding_model.embed_query(search_prompt))
        cluster_relevances = {cid: np.dot(query_vector, c_vec) / (np.linalg.norm(query_vector) * np.linalg.norm(c_vec))
                              for cid, c_vec in genlang_centroids.items() if np.linalg.norm(c_vec) > 0}
        if not cluster_relevances: return "Nenhum cluster relevante encontrado."
        top_clusters = sorted(cluster_relevances.items(), key=lambda item: item[1], reverse=True)
        relevant_clusters = [cid for cid, score in top_clusters if score > 0.7][:top_k]
        if not relevant_clusters: return "Nenhum contexto de processo altamente relevante encontrado."
        context_examples = [
            max(self.memory.concept_clusters[cid], key=lambda v: v.self_eval_score, default=None).source_text for cid in
            relevant_clusters]
        return "\n---\n".join(filter(None, context_examples)) or "Nenhum exemplo claro no contexto recuperado."

    def get_response(self, user_input: str) -> str:
        # <<< MUDANÇA CRÍTICA 2: Recupera fatos do armazém global, não por sessão. >>>
        known_facts_str = json.dumps(self.fact_store, indent=2,
                                     ensure_ascii=False) if self.fact_store else "Nenhum fato conhecido ainda."

        history_str = "\n".join([f"User: {turn['user']}\nBot: {turn['bot']}" for turn in self.conversation_history])
        search_prompt = f"Contexto da conversa atual:\n{history_str}\n\nPergunta atual do usuário: {user_input}"
        retrieved_context = self._retrieve_context(search_prompt)

        llm = self._get_llm_instance()
        chain = self.prompt_template | llm | self.parser

        response_model: ChatResponseModel = chain.invoke({
            "known_facts": known_facts_str,
            "retrieved_context": retrieved_context,
            "history": history_str,
            "user_input": user_input,
            "format_instructions": self.parser.get_format_instructions()
        })

        # <<< MUDANÇA CRÍTICA 3: Atualiza o armazém global de fatos. >>>
        if response_model.extracted_facts:
            logger.info(f"FATOS NOVOS EXTRAÍDOS: {response_model.extracted_facts}")
            self.fact_store.update(response_model.extracted_facts)

        # O resto da lógica (governança, memória de processo) permanece o mesmo.
        thought_text = f"Ao ser perguntado '{user_input}', o sistema raciocinou: '{response_model.thought_process}' e respondeu com sucesso avaliado em {response_model.self_evaluation_score:.2f}."
        thought_vector_np = np.array(self.embedding_model.embed_query(response_model.thought_process))
        current_thought = GenlangVector(thought_vector_np, thought_text, "ChatbotAgent",
                                        response_model.self_evaluation_score)
        self.memory.store_concept(current_thought)
        self.governor.record_turn(current_thought, response_model.self_evaluation_score)
        new_temp, cog_state, _ = self.governor.assess_and_govern(self.system_state["llm_temperature"])
        self.system_state["llm_temperature"] = new_temp
        logger.info(f"Estado Cognitivo: {cog_state} | Nova Temperatura: {new_temp:.2f}")

        self.conversation_history.append({'user': user_input, 'bot': response_model.final_response})
        return response_model.final_response

    def start_new_session(self):
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        # A memória de fatos NÃO é resetada.

        vocab_size = self.memory.get_vocabulary_size()
        temp = self.system_state.get("llm_temperature", 0.6)
        print("\n" + "=" * 60)
        print(f"🤖 Nova sessão de conversa iniciada ({self.session_id})")
        print(
            f"🧠 Memória de Processo: {vocab_size} conceitos | 📋 Memória de Fatos do Usuário: {len(self.fact_store)} fatos | 🔥 Temp: {temp:.2f}")
        print("=" * 60)

    # ... run e shutdown permanecem os mesmos ...
    def run(self):
        self.start_new_session()
        while True:
            try:
                user_input = input("Você: ")
                if user_input.lower() in ['sair', 'exit', 'quit']: break
                if user_input.lower() == 'novasessao': self.start_new_session(); continue
                print(f"Bot: {self.get_response(user_input)}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Ocorreu um erro inesperado: {e}", exc_info=True)
                print("🤖 Desculpe, encontrei um problema. Vamos tentar de novo.")
        self.shutdown()

    def shutdown(self):
        logger.info("Encerrando o chatbot..."); self.save_state(self.checkpoint_path); print(
            "\n🤖 Progresso final salvo. Até logo!")

    def save_state(self, filepath: str):
        state = {
            'memory': self.memory,
            'governor': self.governor,
            'fact_store': self.fact_store,  # Salva o armazém de fatos global
            'system_state': self.system_state
        }
        with open(filepath, 'wb') as f: pickle.dump(state, f)
        logger.info(f"🔥 Estado completo (Processo, Fatos, Governador) persistido em: {filepath}")

    def load_state(self, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            self.memory = state['memory']
            self.governor = state['governor']
            # <<< MUDANÇA CRÍTICA 4: Lógica para carregar o armazém de fatos. >>>
            # Lida tanto com o formato antigo (dicionário de dicionários) quanto com o novo (dicionário simples).
            loaded_facts = state.get('fact_store', {})
            if loaded_facts and isinstance(next(iter(loaded_facts.values()), None), dict):
                logger.warning(
                    "Formato antigo de fact_store detectado. Tentando migrar para o novo formato (um único usuário).")
                # Pega os fatos da primeira sessão encontrada e os torna globais.
                self.fact_store = next(iter(loaded_facts.values()), {})
            else:
                self.fact_store = loaded_facts

            self.system_state = state.get('system_state', {"llm_temperature": 0.6})
        logger.info(f"✅ Estado completo carregado de: {filepath}")


if __name__ == "__main__":
    CHECKPOINT_FILE = "genlang_chatbot_final_state.pkl"
    try:
        chatbot = GenlangChatbot(checkpoint_path=CHECKPOINT_FILE)
        chatbot.run()
    except Exception as e:
        logger.error(f"Ocorreu um erro fatal ao iniciar o chatbot: {e}", exc_info=True)
