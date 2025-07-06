import os
import numpy as np
import json
import logging
from typing import List, Tuple, Dict, Optional, TypedDict
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import random
import pickle  # <-- NOVO: Import para salvar/carregar estado

# LangChain / LangGraph Imports
# ... (o resto das importações permanece o mesmo) ...
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field as PydanticField
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END

# Carregar variáveis de ambiente (para a API key)
from dotenv import load_dotenv

load_dotenv()

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================
# ESTRUTURAS DE DADOS BASE
# ... (sem mudanças aqui) ...
# ==========================
@dataclass
class GenlangVector:
    vector: np.ndarray
    source_text: str
    source_agent: str

    def similarity(self, other: 'GenlangVector') -> float:
        cos_sim = np.dot(self.vector, other.vector) / (np.linalg.norm(self.vector) * np.linalg.norm(other.vector))
        return float(cos_sim)


@dataclass
class MathTask:
    task_id: int;
    task_type: str;
    problem_statement: str;
    expected_output: float;
    complexity: int = 1


class CognitiveState(Enum):
    RIGID = "rigid";
    OPTIMAL = "optimal";
    CHAOTIC = "chaotic"


# ==========================
# CAMADA 1: AGENTES COMUNICADORES
# ... (sem mudanças aqui) ...
# ==========================
class LLMGenlangAgent:
    def __init__(self, agent_id: str, specialization: str, embedding_model: OpenAIEmbeddings,
                 vector_modification_ratio: float = 0.5):
        self.agent_id = agent_id;
        self.specialization = specialization;
        self.embedding_model = embedding_model
        self.vector_modification_ratio = vector_modification_ratio;
        self.llm: Optional[ChatOpenAI] = None
        self.creation_prompt = ChatPromptTemplate.from_messages([("system",
                                                                  "Você é o Agente {agent_id}, especialista em {specialization}. Gere um pensamento curto e conciso para o próximo passo na solução do problema matemático, baseado no histórico."),
                                                                 ("human",
                                                                  "Problema: '{problem_statement}'\nHistórico:\n{conversation_history}\n\nSeu pensamento:")])
        self.modification_prompt = ChatPromptTemplate.from_messages([("system",
                                                                      "Você é um controlador de pensamento vetorial. Descreva verbalmente a *modificação* que o vetor de pensamento atual precisa para se aproximar da solução. Use termos como 'adicionar conceito X', 'aumentar importância Y', 'correlacionar com operação Z'."),
                                                                     ("human",
                                                                      "Problema: '{problem_statement}'\nContexto Atual: '{context_summary}'\n\nSua descrição da modificação:")])

    def set_llm(self, llm: ChatOpenAI):
        self.llm = llm

    def generate_concept(self, task: MathTask, context: List[GenlangVector]) -> GenlangVector:
        if self.llm is None: raise ValueError("LLM não foi definido para o agente. Chame set_llm() primeiro.")
        should_modify = context and random.random() < self.vector_modification_ratio
        if should_modify:
            context_summary = " -> ".join([c.source_text for c in context[-3:]]);
            context_vector = np.mean([c.vector for c in context], axis=0)
            chain = self.modification_prompt | self.llm;
            response = chain.invoke({"problem_statement": task.problem_statement, "context_summary": context_summary})
            modification_text = response.content;
            delta_vector = np.array(self.embedding_model.embed_query(modification_text))
            new_vector = context_vector + delta_vector;
            new_vector /= np.linalg.norm(new_vector)
            source_text = f"[MOD] {modification_text}";
            logger.debug(f"Agente {self.agent_id} modificou o pensamento com: '{modification_text}'")
            return GenlangVector(vector=new_vector, source_text=source_text, source_agent=self.agent_id)
        else:
            history_str = "\n".join(
                [f"- {c.source_agent}: '{c.source_text}'" for c in context]) or "(Nenhuma conversa ainda)";
            chain = self.creation_prompt | self.llm
            response = chain.invoke({"agent_id": self.agent_id, "specialization": self.specialization,
                                     "problem_statement": task.problem_statement, "conversation_history": history_str})
            thought_text = response.content;
            vector = np.array(self.embedding_model.embed_query(thought_text));
            logger.debug(f"Agente {self.agent_id} criou o pensamento: '{thought_text}'")
            return GenlangVector(vector=vector, source_text=thought_text, source_agent=self.agent_id)


# ==========================
# CAMADA 2: GOVERNADOR DE CRITICALIDADE
# ==========================
class CriticalityGovernor:
    def __init__(self, history_size: int = 100):
        self.communication_history = deque(maxlen=history_size);
        self.task_success_history = deque(maxlen=history_size)

    # --- MODIFICADO: Agora aceita um float para o sucesso ponderado ---
    def record_cycle(self, exchanges: List[GenlangVector], weighted_success: float):
        self.communication_history.extend(exchanges)
        self.task_success_history.append(weighted_success)

    def assess_and_govern(self, current_temp: float) -> Tuple[float, CognitiveState, Dict[str, float]]:
        if len(self.communication_history) < 20: return current_temp, CognitiveState.OPTIMAL, {"novelty": 0.5,
                                                                                               "coherence": 0.5,
                                                                                               "grounding": 0.5}
        metrics = {"novelty": self._calculate_novelty(), "coherence": self._calculate_coherence(),
                   "grounding": self._calculate_grounding()}
        new_temp = current_temp
        if metrics["novelty"] < 0.3 and metrics["coherence"] > 0.6:
            state = CognitiveState.RIGID;
            new_temp = min(current_temp * 1.2, 1.3);
            logger.warning(f"Estado RÍGIDO detectado. Aumentando temperatura para {new_temp:.2f}")
        elif metrics["novelty"] > 0.7 and metrics["coherence"] < 0.3:
            state = CognitiveState.CHAOTIC;
            new_temp = max(current_temp * 0.8, 0.1);
            logger.warning(f"Estado CAÓTICO detectado. Reduzindo temperatura para {new_temp:.2f}")
        else:
            state = CognitiveState.OPTIMAL
        return new_temp, state, metrics

    def _calculate_novelty(self) -> float:
        if len(self.communication_history) < 20: return 0.5
        recent, older = list(self.communication_history)[-10:], list(self.communication_history)[:-10]
        if not older: return 0.8
        distances = [1 - r.similarity(o) for r in recent for o in older];
        return np.mean(distances) if distances else 0.5

    def _calculate_coherence(self) -> float:
        if len(self.communication_history) < 2: return 0.5
        recent = list(self.communication_history)[-20:]
        similarities = [recent[i].similarity(recent[i + 1]) for i in range(len(recent) - 1)];
        return np.mean(similarities) if similarities else 0.5

    def _calculate_grounding(self) -> float:
        if not self.task_success_history: return 0.5
        return np.mean(list(self.task_success_history)[-50:])


# ==========================
# CAMADA 3: MEMÓRIA E META-APRENDIZADO
# ... (sem mudanças aqui) ...
# ==========================
class MemorySystem:
    def __init__(self, cluster_threshold: float = 0.8):
        self.concept_clusters: Dict[str, List[GenlangVector]] = {};
        self.cluster_threshold = cluster_threshold

    def store_communication(self, exchange: List[GenlangVector]):
        for vector in exchange: self._update_concept_clusters(vector)

    def _update_concept_clusters(self, vector: GenlangVector):
        if not self.concept_clusters: self.concept_clusters["concept_0"] = [vector]; return
        similarities = {cid: np.mean([vector.similarity(v) for v in c_vectors]) for cid, c_vectors in
                        self.concept_clusters.items()}
        best_cluster, best_sim = max(similarities.items(), key=lambda item: item[1])
        if best_sim > self.cluster_threshold:
            self.concept_clusters[best_cluster].append(vector)
        else:
            self.concept_clusters[f"concept_{len(self.concept_clusters)}"] = [vector]

    def get_vocabulary_size(self) -> int:
        return len(self.concept_clusters)

    def get_cluster_centroids(self) -> Dict[str, np.ndarray]:
        return {cid: np.mean([v.vector for v in vectors], axis=0) for cid, vectors in self.concept_clusters.items() if
                vectors}


# ==========================
# SISTEMA PRINCIPAL COM LANGGRAPH E GOVERNANÇA ATIVA
# ==========================

# --- MODIFICADO: GraphState agora inclui weighted_success ---
class GraphState(TypedDict):
    task: MathTask;
    exchanges: List[GenlangVector];
    current_agent_idx: int;
    max_exchanges: int
    llm_temperature: float;
    solution: Optional[float];
    success: bool
    weighted_success: Optional[float];
    error: Optional[float]


class EmergentGenlangSystem:
    def __init__(self, vector_dim: int = 256, max_exchanges: int = 8, initial_temp: float = 0.5):
        self.max_exchanges = max_exchanges;
        self.initial_temp = initial_temp
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=vector_dim)
        self.agents = [LLMGenlangAgent("Agente_A", "Analista Estratégico", self.embedding_model),
                       LLMGenlangAgent("Agente_B", "Executor Lógico", self.embedding_model),
                       LLMGenlangAgent("Agente_C", "Verificador de Estado", self.embedding_model)]
        self.governor = CriticalityGovernor();
        self.memory = MemorySystem()
        self.task_history: List[Dict] = [];
        self.metrics_history: List[Dict] = [];
        self.centroid_history: List[Dict] = []
        self.graph = self._build_graph()

    # --- NOVO: Métodos para Salvar e Carregar o "Cérebro" do Sistema ---
    def save_state(self, filepath: str):
        """Salva o estado da memória, governador e históricos em um arquivo."""
        state = {
            'memory': self.memory,
            'governor': self.governor,
            'task_history': self.task_history,
            'metrics_history': self.metrics_history,
            'centroid_history': self.centroid_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Estado do sistema salvo em {filepath}")

    def load_state(self, filepath: str):
        """Carrega o estado da memória, governador e históricos de um arquivo."""
        if not os.path.exists(filepath):
            logger.warning(f"Arquivo de checkpoint '{filepath}' não encontrado. Iniciando com estado limpo.")
            return
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.memory = state['memory']
            self.governor = state['governor']
            self.task_history = state.get('task_history', [])
            self.metrics_history = state.get('metrics_history', [])
            self.centroid_history = state.get('centroid_history', [])
            logger.info(
                f"Estado do sistema carregado de '{filepath}'. Vocabulário atual: {self.memory.get_vocabulary_size()}.")
        except Exception as e:
            logger.error(f"Falha ao carregar o estado de '{filepath}': {e}. Iniciando com estado limpo.")

    def _get_llm_instance(self, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(model="gpt-4.1-nano", temperature=temperature)

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        def agent_node(state: GraphState) -> GraphState:
            agent = self.agents[state["current_agent_idx"]];
            llm_instance = self._get_llm_instance(state["llm_temperature"]);
            agent.set_llm(llm_instance)
            concept = agent.generate_concept(state["task"], state["exchanges"]);
            state["exchanges"].append(concept)
            state["current_agent_idx"] = (state["current_agent_idx"] + 1) % len(self.agents);
            return state

        def interpreter_node(state: GraphState) -> GraphState:
            conversation_str = "\n".join([f"- {c.source_agent}: '{c.source_text}'" for c in state["exchanges"]])

            class NumericSolution(BaseModel):
                answer: Optional[float] = PydanticField(
                    description="A resposta numérica final. Use null se não houver.")

            parser = PydanticOutputParser(pydantic_object=NumericSolution)
            prompt = ChatPromptTemplate.from_template(
                "Analise a conversa para resolver o problema. Extraia a resposta numérica final em JSON.\nProblema: {problem}\nConversa:\n{conversation}\n{format_instructions}")
            chain = prompt | self._get_llm_instance(0.1) | parser
            try:
                result = chain.invoke({"problem": state["task"].problem_statement, "conversation": conversation_str,
                                       "format_instructions": parser.get_format_instructions()})
                solution = result.answer
            except Exception:
                solution = None

            tolerance = 0.1 if state['task'].task_type != 'physics' else 0.5
            is_correct = solution is not None and abs(solution - state["task"].expected_output) < tolerance

            # --- NOVO: Mecanismo de Pressão por Compressão ---
            num_exchanges = len(state["exchanges"])
            efficiency_penalty = max(0, (num_exchanges - 1) * 0.1)  # Penalidade de 10% por troca extra além da primeira
            weighted_success = (1.0 - efficiency_penalty) if is_correct else 0.0

            state["success"] = is_correct
            state["weighted_success"] = weighted_success
            state["solution"] = solution
            state["error"] = abs(solution - state["task"].expected_output) if solution is not None else None
            return state

        def should_continue(state: GraphState) -> str:
            if len(state["exchanges"]) >= self.max_exchanges: logger.warning(
                f"Tarefa {state['task'].task_id} atingiu o máximo de trocas.")
            return "end" if state["success"] or len(state["exchanges"]) >= self.max_exchanges else "continue"

        workflow.add_node("agent", agent_node);
        workflow.add_node("interpreter", interpreter_node)
        workflow.set_entry_point("agent");
        workflow.add_edge("agent", "interpreter")
        workflow.add_conditional_edges("interpreter", should_continue, {"continue": "agent", "end": END})

        return workflow.compile()

    def train(self, num_epochs: int = 20, tasks_per_epoch: int = 10, start_epoch: int = 0,
              force_complexity: Optional[int] = None, log_filename: str = "training_log.jsonl"):
        epoch_state = {"llm_temperature": self.initial_temp}

        for epoch_idx in range(start_epoch, start_epoch + num_epochs):
            complexity = force_complexity if force_complexity is not None else min(epoch_idx // 10 + 1, 10)

            for task_idx in range(tasks_per_epoch):
                task_id = epoch_idx * tasks_per_epoch + task_idx
                task = self._generate_task(task_id, complexity)
                logger.info(
                    f"Epoch {epoch_idx + 1}, Cmplx {complexity}, Task {task_idx + 1}: [{task.task_type}] {task.problem_statement}")

                initial_state = GraphState(task=task, exchanges=[], current_agent_idx=0,
                                           max_exchanges=self.max_exchanges,
                                           llm_temperature=epoch_state["llm_temperature"], solution=None, success=False,
                                           weighted_success=None, error=None)
                config = {"recursion_limit": self.max_exchanges * 2 + 5}
                final_state = self.graph.invoke(initial_state, config=config)

                # --- NOVO: Logging Estruturado ---
                log_entry = {"task_id": task.task_id, "epoch": epoch_idx, "complexity": complexity,
                             "task_type": task.task_type, "problem": task.problem_statement,
                             "expected_output": task.expected_output, "success": final_state['success'],
                             "weighted_success": final_state['weighted_success'], "solution": final_state['solution'],
                             "exchanges_count": len(final_state['exchanges']),
                             "conversation": [{"agent": ex.source_agent, "thought": ex.source_text} for ex in
                                              final_state['exchanges']]}
                with open(log_filename, "a", encoding="utf-8") as f: f.write(
                    json.dumps(log_entry, ensure_ascii=False) + "\n")

                # --- MODIFICADO: Passa o sucesso ponderado para o Governor ---
                self.governor.record_cycle(final_state['exchanges'], final_state.get('weighted_success', 0.0))
                self.task_history.append(
                    {'epoch': epoch_idx, 'success': final_state['success'], 'exchanges': len(final_state['exchanges']),
                     'task_type': task.task_type, 'weighted_success': final_state.get('weighted_success', 0.0)})
                self.memory.store_communication(final_state['exchanges'])

            new_temp, cog_state, metrics = self.governor.assess_and_govern(epoch_state["llm_temperature"])
            epoch_state["llm_temperature"] = new_temp
            self.metrics_history.append({**metrics, 'epoch': epoch_idx, 'temperature': new_temp})
            self.centroid_history.append({'epoch': epoch_idx, 'centroids': self.memory.get_cluster_centroids()})

            success_rate = np.mean([h['success'] for h in self.task_history if h['epoch'] == epoch_idx])
            logger.info(
                f"Epoch {epoch_idx + 1} Resumo: Sucesso: {success_rate:.2f}, Vocabulário: {self.memory.get_vocabulary_size()}, Estado: {cog_state.value}, Temp: {new_temp:.2f}")

            # --- NOVO: Salvamento periódico de checkpoints ---
            if (epoch_idx + 1) % 20 == 0:
                self.save_state(f"genlang_state_epoch_{epoch_idx + 1}.pkl")

    # ... (plot_results e geradores de tarefas permanecem os mesmos) ...
    def plot_results(self):
        if not self.task_history: logger.warning("Nenhum histórico para plotar."); return
        df_tasks = pd.DataFrame(self.task_history);
        df_metrics = pd.DataFrame(self.metrics_history)
        success_by_type = df_tasks.groupby('task_type')['success'].mean()
        logger.info(f"Taxa de sucesso por tipo de tarefa:\n{success_by_type}")
        fig, axes = plt.subplots(4, 1, figsize=(14, 22), sharex=True)
        fig.suptitle("Análise da Evolução do Sistema Genlang", fontsize=16)
        df_epoch_summary = df_tasks.groupby('epoch')['success'].mean().reset_index()
        axes[0].plot(df_epoch_summary['epoch'], df_epoch_summary['success'], 'o-', label='Taxa de Sucesso (Booleano)',
                     alpha=0.7)
        axes[0].plot(df_epoch_summary['epoch'], df_epoch_summary['success'].rolling(window=10, min_periods=1).mean(),
                     'r-', label='Média Móvel (10 épocas)')
        axes[0].set_ylabel("Taxa de Sucesso");
        axes[0].set_title("Desempenho da Tarefa");
        axes[0].grid(True);
        axes[0].legend();
        axes[0].set_ylim(-0.1, 1.1)
        ax2_twin = axes[1].twinx()
        df_vocab = pd.Series({h['epoch']: len(h['centroids']) for h in self.centroid_history}).reset_index(
            name='vocab_size').rename(columns={'index': 'epoch'})
        df_efficiency = df_tasks.groupby('epoch')['exchanges'].mean().reset_index()
        axes[1].plot(df_vocab['epoch'], df_vocab['vocab_size'], 'g-s', label='Tamanho do Vocabulário');
        ax2_twin.plot(df_efficiency['epoch'], df_efficiency['exchanges'], 'm-p', label='Média de Trocas (Eficiência)')
        axes[1].set_ylabel("Nº de Conceitos", color='g');
        ax2_twin.set_ylabel("Nº de Trocas", color='m');
        axes[1].set_title("Vocabulário Emergente e Eficiência Comunicativa");
        axes[1].grid(True)
        lines1, labels1 = axes[1].get_legend_handles_labels();
        lines2, labels2 = ax2_twin.get_legend_handles_labels();
        ax2_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
        ax3_twin = axes[2].twinx()
        axes[2].plot(df_metrics['epoch'], df_metrics['novelty'], '-o', label='Novelty');
        axes[2].plot(df_metrics['epoch'], df_metrics['coherence'], '-s', label='Coherence');
        axes[2].plot(df_metrics['epoch'], df_metrics['grounding'], '-^', label='Grounding (Eficiência)')
        ax3_twin.plot(df_metrics['epoch'], df_metrics['temperature'], ':r', label='LLM Temp');
        axes[2].set_ylabel("Valor da Métrica");
        ax3_twin.set_ylabel("Temperatura", color='r');
        axes[2].set_title("Evolução do Estado Cognitivo e Governança");
        axes[2].grid(True)
        lines1, labels1 = axes[2].get_legend_handles_labels();
        lines2, labels2 = ax3_twin.get_legend_handles_labels();
        ax3_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
        drifts = [];
        epochs_for_drift = []
        if len(self.centroid_history) > 1:
            for i in range(1, len(self.centroid_history)):
                prev_c, curr_c = self.centroid_history[i - 1]['centroids'], self.centroid_history[i]['centroids']
                common_keys = set(prev_c.keys()) & set(curr_c.keys())
                epoch_drift = [np.linalg.norm(prev_c[k] - curr_c[k]) for k in common_keys]
                if epoch_drift: drifts.append(np.mean(epoch_drift)); epochs_for_drift.append(
                    self.centroid_history[i]['epoch'])
        if drifts: axes[3].plot(epochs_for_drift, drifts, '-x', color='purple', label='Instabilidade Conceitual')
        axes[3].set_xlabel("Época");
        axes[3].set_ylabel("Mudança Média do Centroide");
        axes[3].set_title("Estabilidade da Linguagem Emergente");
        axes[3].grid(True);
        axes[3].legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]);
        plt.show()

    def _generate_arithmetic_task(self, complexity: int) -> Tuple[str, float]:
        a = np.random.randint(1, 10 * complexity);
        b = np.random.randint(1, 10 * complexity)
        op = random.choice(['+', '-', '*', '/']);
        b = 1 if op == '/' and b == 0 else b
        problem = f"{a} {op} {b}";
        result = eval(problem);
        return f"Calcule: {problem}", float(result)

    def _generate_algebra_task(self, complexity: int) -> Tuple[str, float]:
        a = np.random.randint(1, 5 * complexity);
        x = np.random.randint(-10, 10);
        b = np.random.randint(-20, 20);
        c = a * x + b
        return f"Resolva para x: {a}x + {b} = {c}", float(x)

    def _generate_noisy_task(self, complexity: int) -> Tuple[str, float]:
        a = np.random.randint(1, 10 * complexity);
        b = np.random.randint(1, 10 * complexity);
        result = a + b
        num_wagons = random.randint(5, 20);
        temperature = random.randint(0, 30);
        color = random.choice(["azul", "vermelho", "verde"])
        return (
            f"Um trem com {num_wagons} vagões {color} transporta {a} passageiros. Mais {b} passageiros embarcam. A temperatura é {temperature}°C. Total de passageiros?"), float(
            result)

    def _generate_multistep_logic_task(self, complexity: int) -> Tuple[str, float]:
        b = random.randint(1, 5 * complexity);
        c = random.randint(1, 5 * complexity);
        e = random.randint(2, 4);
        a = b + c;
        d = a * e
        return (f"A é a soma de {b} e {c}. O resultado é A multiplicado por {e}. Qual o resultado?"), float(d)

    def _generate_fibonacci_task(self, complexity: int) -> Tuple[str, float]:
        n = random.randint(5, 8 + complexity)


        def fib(num: int) -> int:
            """Calcula o n-ésimo número de Fibonacci de forma iterativa."""
            a, b = 0, 1

            for _ in range(num):

                a, b = b, a + b

            return a

        result = fib(n)
        return (f"Na sequência de Fibonacci onde F(0)=0 e F(1)=1, qual é o valor de F({n})?"), float(result)

    def _generate_physics_task(self, complexity: int) -> Tuple[str, float]:
        pos = 0;
        vel = random.randint(5, 10 + complexity);
        friction = random.randint(1, 3);
        time_steps = random.randint(3, 5)
        final_pos = pos;
        current_vel = vel
        for _ in range(time_steps): final_pos += current_vel; current_vel = max(0, current_vel - friction)
        return (
            f"Uma bola está na posição {pos} com velocidade {vel}. A cada segundo, sua velocidade diminui em {friction}. Qual a posição após {time_steps} segundos?"), float(
            final_pos)

    def _generate_task(self, task_id: int, complexity: int) -> MathTask:
        task_pool = {'arithmetic': self._generate_arithmetic_task, 'algebra': self._generate_algebra_task}
        if complexity >= 2: task_pool['noisy_logic'] = self._generate_noisy_task
        if complexity >= 3: task_pool['multistep_logic'] = self._generate_multistep_logic_task
        if complexity >= 5: task_pool['recursion'] = self._generate_fibonacci_task
        if complexity >= 7: task_pool['physics'] = self._generate_physics_task
        task_type_str, task_func = random.choice(list(task_pool.items()));
        problem_statement, expected_output = task_func(complexity)
        return MathTask(task_id=task_id, task_type=task_type_str, problem_statement=problem_statement,
                        expected_output=expected_output, complexity=complexity)


# ==========================
# DEMONSTRAÇÃO
# ==========================
if __name__ == "__main__":

    # --- Cenário 1: Treinamento Base do Zero ---

     logger.info("Cenário 1: Iniciando Treinamento Base do Zero...")
     system_base = EmergentGenlangSystem(vector_dim=256, max_exchanges=15, initial_temp=0.7)
     system_base.train(num_epochs=300, tasks_per_epoch=30)
     logger.info("\nTreinamento Base concluído.")
     system_base.plot_results()
     system_base.save_state("final_base_model_epoch_300.pkl")



