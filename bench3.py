# bench3_refactored.py
import os
import sys
import pickle
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import logging
import random
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# --- Importar as classes do seu sistema para desserialização ---
try:
    from code import MemorySystem, GenlangVector, CognitiveState, CriticalityGovernor, EmergentGenlangSystem
except ImportError:
    print("AVISO: 'code.py' não encontrado. Definindo classes dummy para desserialização.")


    class MemorySystem:
        def __init__(self):
            self.concept_clusters = {}

        def get_cluster_centroids(self):
            return {}


    class EmergentGenlangSystem:
        def __init__(self, **kwargs):
            pass

        def _generate_task(self, task_id, complexity):
            return type('Task', (), {
                'problem_statement': 'Dummy problem',
                'expected_output': 42.0,
                'task_type': 'dummy'
            })()


    class GenlangVector:
        def __init__(self, vector=None, source_text="", source_agent=""):
            self.vector = vector or np.zeros(256)
            self.source_text = source_text
            self.source_agent = source_agent


    class CognitiveState:
        pass


    class CriticalityGovernor:
        pass

# --- Configurações ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Modelo de Saída para extrair a resposta numérica
class NumericSolution(BaseModel):
    answer: Optional[float] = Field(description="A resposta numérica final. Use null se não for possível determinar.")


# ===============================================
# GERADORES DE TESTE ALINHADOS COM TREINAMENTO
# ===============================================

class AlignedTestGenerator:
    """Gera problemas alinhados com o que o Genlang foi treinado"""

    def __init__(self, genlang_state):
        self.genlang_state = genlang_state
        self.task_history = genlang_state.get('task_history', [])

    def analyze_training_distribution(self):
        """Analisa a distribuição de problemas no treinamento"""
        type_counts = defaultdict(int)
        type_success = defaultdict(list)

        for task in self.task_history:
            task_type = task.get('task_type', 'unknown')
            type_counts[task_type] += 1
            type_success[task_type].append(task.get('success', False))

        return {
            task_type: {
                'count': count,
                'success_rate': np.mean(type_success[task_type])
            }
            for task_type, count in type_counts.items()
        }

    def generate_in_distribution_tests(self, num_per_type: int = 5) -> List[Dict]:
        """Gera testes similares ao que foi treinado"""
        tests = []

        # Física com parâmetros similares aos do treinamento
        for i in range(num_per_type):
            v = random.randint(5, 15)
            a = random.randint(1, 4)
            t = random.randint(3, 7)
            tests.append({
                'problem': f"Uma bola está na posição 0 com velocidade {v} e atrito {a}. Qual a posição após {t} segundos?",
                'expected': self._calculate_physics_position(v, a, t),
                'type': 'physics',
                'category': 'in_distribution'
            })

        # Álgebra no estilo do treinamento
        for i in range(num_per_type):
            a = random.randint(2, 30)
            b = random.randint(-20, 20)
            x = random.randint(-10, 10)
            c = a * x + b
            tests.append({
                'problem': f"Resolva para x: {a}x + {b} = {c}",
                'expected': float(x),
                'type': 'algebra',
                'category': 'in_distribution'
            })

        # Multi-step logic
        for i in range(num_per_type):
            b = random.randint(10, 50)
            c = random.randint(10, 50)
            e = random.randint(2, 4)
            tests.append({
                'problem': f"A é a soma de {b} e {c}. O resultado é A multiplicado por {e}. Qual o resultado?",
                'expected': float((b + c) * e),
                'type': 'multistep_logic',
                'category': 'in_distribution'
            })

        return tests

    def generate_slight_variations(self, num_per_type: int = 3) -> List[Dict]:
        """Gera variações sutis dos problemas treinados"""
        variations = []

        # Vocabulário diferente para física
        variations.extend([
            {
                'problem': "Object at x=0, initial speed 10, friction coefficient 2. Find position at t=5.",
                'expected': 25.0,
                'type': 'physics',
                'category': 'variation_vocabulary'
            },
            {
                'problem': "v₀=12, a=-3, calcule x(4)",
                'expected': 24.0,
                'type': 'physics',
                'category': 'variation_notation'
            }
        ])

        # Ordem diferente em multi-step
        variations.append({
            'problem': "Multiplique 3 pelo resultado da soma de 15 com 20. Qual é o valor?",
            'expected': 105.0,
            'type': 'multistep_logic',
            'category': 'variation_order'
        })

        return variations

    def generate_edge_cases(self) -> List[Dict]:
        """Gera casos extremos mas ainda dentro do domínio"""
        edge_cases = []

        # Física com valores extremos
        edge_cases.append({
            'problem': "Bola com velocidade 0 e atrito 5. Posição após 10 segundos?",
            'expected': 0.0,
            'type': 'physics',
            'category': 'edge_zero_velocity'
        })

        # Álgebra com solução negativa
        edge_cases.append({
            'problem': "Resolva para x: -5x + 10 = 25",
            'expected': -3.0,
            'type': 'algebra',
            'category': 'edge_negative'
        })

        return edge_cases

    def _calculate_physics_position(self, v0: float, friction: float, time: int) -> float:
        """Calcula posição usando a mesma lógica do treinamento"""
        pos = 0
        vel = v0
        for _ in range(time):
            pos += vel
            vel = max(0, vel - friction)
        return pos


# ===============================================
# BENCHMARK FOCADO EM DOMÍNIO - COMPLETO
# ===============================================

class DomainFocusedBenchmark:
    """Benchmark que testa o Genlang no que ele foi projetado para fazer"""

    def __init__(self, epoch_checkpoint_path: str):
        logger.info(f"Carregando Genlang de: {epoch_checkpoint_path}")
        with open(epoch_checkpoint_path, 'rb') as f:
            self.adaptive_state = pickle.load(f)

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=256)
        self.output_parser = PydanticOutputParser(pydantic_object=NumericSolution)

        # Analisador de distribuição
        self.test_generator = AlignedTestGenerator(self.adaptive_state)

        # Prepara centroids do Genlang
        self.genlang_centroids = self.adaptive_state['memory'].get_cluster_centroids()

        # Prepara RAG tradicional
        self._prepare_traditional_rag()

    def _prepare_traditional_rag(self):
        """Prepara RAG tradicional com os mesmos dados"""
        try:
            all_texts = []
            for cluster in self.adaptive_state['memory'].concept_clusters.values():
                for vector in cluster:
                    all_texts.append(vector.source_text)

            splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
            chunks = splitter.create_documents(all_texts)

            self.traditional_vectorstore = FAISS.from_documents(chunks, self.embeddings)
            self.traditional_retriever = self.traditional_vectorstore.as_retriever(search_kwargs={"k": 5})
            logger.info("RAG tradicional preparado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao preparar RAG tradicional: {e}")
            # Criar um mock retriever
            self.traditional_retriever = None

    def _extract_solution(self, text: str) -> Optional[float]:
        """Extrai a resposta numérica de uma string de forma robusta"""
        try:
            return self.output_parser.parse(text).answer
        except Exception:
            # Fallback: extrair números da resposta
            matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
            return float(matches[-1]) if matches else None

    def solve_with_genlang(self, problem: str) -> Dict[str, Any]:
        """Resolve usando a memória conceitual do Genlang"""
        start_time = time.time()

        try:
            # 1. Busca: Encontra os clusters mais relevantes baseados nos centróides
            query_vector = np.array(self.embeddings.embed_query(problem))
            cluster_relevances = {}

            for cid, c_vec in self.genlang_centroids.items():
                similarity = np.dot(query_vector, c_vec) / (np.linalg.norm(query_vector) * np.linalg.norm(c_vec))
                cluster_relevances[cid] = similarity

            # Pega os 3 clusters mais relevantes
            top_clusters = sorted(cluster_relevances.items(), key=lambda item: item[1], reverse=True)[:3]

            # 2. Geração de Contexto: Extrai exemplos dos melhores clusters
            context_examples = []
            for cid, _ in top_clusters:
                cluster_vectors = self.adaptive_state['memory'].concept_clusters[cid]
                # Pega até 2 exemplos de cada cluster
                examples_from_cluster = cluster_vectors[:2]
                context_examples.extend([v.source_text for v in examples_from_cluster])

            context_str = "\n---\n".join(list(set(context_examples)))

            # 3. Invocação do LLM com o contexto especializado
            prompt = ChatPromptTemplate.from_template(
                "Use os seguintes exemplos de uma memória conceitual para resolver o problema.\n"
                "Exemplos de Conceitos Relevantes:\n{context}\n\n"
                "Problema Original: {problem}\n\n"
                "Resposta Final (formato JSON):\n{format_instructions}"
            )
            chain = prompt | self.llm | StrOutputParser()
            response_text = chain.invoke({
                "context": context_str,
                "problem": problem,
                "format_instructions": self.output_parser.get_format_instructions()
            })

            solution = self._extract_solution(response_text)
        except Exception as e:
            logger.error(f"Erro no solver Genlang: {e}")
            solution = None
            context_str = "Erro ao recuperar contexto"

        end_time = time.time()

        return {
            "solution": solution,
            "time": end_time - start_time,
            "context_source": "Genlang Clusters",
            "num_context_items": len(top_clusters) if 'top_clusters' in locals() else 0,
            "retrieved_context": context_str
        }

    def solve_with_traditional_rag(self, problem: str) -> Dict[str, Any]:
        """Resolve usando o RAG tradicional"""
        start_time = time.time()

        try:
            if self.traditional_retriever is None:
                raise Exception("RAG tradicional não disponível")

            # 1. Busca: Recupera os 5 chunks mais similares
            retrieved_docs = self.traditional_retriever.invoke(problem)
            context_str = "\n---\n".join([doc.page_content for doc in retrieved_docs])

            # 2. Invocação do LLM
            prompt = ChatPromptTemplate.from_template(
                "Use os seguintes trechos de texto para resolver o problema.\n"
                "Contexto:\n{context}\n\n"
                "Problema: {problem}\n\n"
                "Resposta Final (formato JSON):\n{format_instructions}"
            )
            chain = prompt | self.llm | StrOutputParser()
            response_text = chain.invoke({
                "context": context_str,
                "problem": problem,
                "format_instructions": self.output_parser.get_format_instructions()
            })

            solution = self._extract_solution(response_text)
        except Exception as e:
            logger.error(f"Erro no RAG tradicional: {e}")
            solution = None
            context_str = "Erro no RAG tradicional"
            retrieved_docs = []

        end_time = time.time()

        return {
            "solution": solution,
            "time": end_time - start_time,
            "context_source": "Traditional RAG",
            "num_context_items": len(retrieved_docs) if 'retrieved_docs' in locals() else 0,
            "retrieved_context": context_str
        }

    def solve_with_pure_llm(self, problem: str) -> Dict[str, Any]:
        """Resolve usando apenas o LLM, sem RAG"""
        start_time = time.time()

        try:
            prompt = ChatPromptTemplate.from_template(
                "Resolva o seguinte problema passo a passo.\n"
                "Problema: {problem}\n\n"
                "Resposta Final (formato JSON):\n{format_instructions}"
            )
            chain = prompt | self.llm | StrOutputParser()
            response_text = chain.invoke({
                "problem": problem,
                "format_instructions": self.output_parser.get_format_instructions()
            })

            solution = self._extract_solution(response_text)
        except Exception as e:
            logger.error(f"Erro no LLM puro: {e}")
            solution = None

        end_time = time.time()

        return {
            "solution": solution,
            "time": end_time - start_time,
            "context_source": "Pure LLM",
            "num_context_items": 0,
            "retrieved_context": "N/A"
        }

    def solve_with_genlang_fast(self, problem: str) -> Dict[str, Any]:
        """Versão otimizada do solver Genlang para testes em lote"""
        try:
            query_vector = np.array(self.embeddings.embed_query(problem))

            # Top cluster apenas
            best_cluster = None
            best_similarity = -1

            for cid, centroid in self.genlang_centroids.items():
                similarity = np.dot(query_vector, centroid) / (np.linalg.norm(query_vector) * np.linalg.norm(centroid))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cid

            if best_cluster is not None:
                # Usa apenas primeiro exemplo
                example = self.adaptive_state['memory'].concept_clusters[best_cluster][0].source_text

                # Resposta rápida
                prompt = f"Baseado em: {example}\nResolva: {problem}\nResposta numérica:"
                response = self.llm.predict(prompt)
                solution = self._extract_number(response)
            else:
                solution = None

        except Exception as e:
            logger.error(f"Erro no Genlang fast: {e}")
            solution = None

        return {'solution': solution}

    def _extract_number(self, text: str) -> Optional[float]:
        """Extração rápida de número"""
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return float(numbers[-1]) if numbers else None

    def test_consistency(self, num_runs: int = 10) -> Dict[str, List[float]]:
        """Testa consistência rodando o mesmo problema múltiplas vezes"""
        logger.info(f"Testando consistência com {num_runs} execuções...")

        test_problem = "Uma bola está na posição 0 com velocidade 10 e atrito 2. Qual a posição após 5 segundos?"
        expected = 30.0

        results = {
            'genlang': [],
            'traditional_rag': [],
            'pure_llm': []
        }

        for i in range(num_runs):
            # Genlang
            sol = self.solve_with_genlang(test_problem)
            if sol['solution'] is not None:
                results['genlang'].append(abs(sol['solution'] - expected))

            # Traditional RAG
            sol = self.solve_with_traditional_rag(test_problem)
            if sol['solution'] is not None:
                results['traditional_rag'].append(abs(sol['solution'] - expected))

            # Pure LLM
            sol = self.solve_with_pure_llm(test_problem)
            if sol['solution'] is not None:
                results['pure_llm'].append(abs(sol['solution'] - expected))

        return results

    def test_batch_efficiency(self, num_problems: int = 100) -> Dict[str, float]:
        """Testa eficiência em lote com muitos problemas similares"""
        logger.info(f"Testando eficiência em lote com {num_problems} problemas...")

        # Gera lote de problemas similares
        problems = []
        for _ in range(num_problems):
            v = random.randint(8, 12)
            a = random.randint(2, 3)
            t = random.randint(4, 6)
            problems.append(f"Velocidade {v}, atrito {a}, posição após {t} segundos?")

        times = {}

        # Teste Genlang
        start = time.time()
        for p in problems:
            self.solve_with_genlang_fast(p)
        times['genlang'] = time.time() - start

        # Teste Traditional RAG
        start = time.time()
        for p in problems:
            self.solve_with_traditional_rag(p)
        times['traditional_rag'] = time.time() - start

        # Teste Pure LLM
        start = time.time()
        for p in problems:
            self.solve_with_pure_llm(p)
        times['pure_llm'] = time.time() - start

        return times

    def test_retrieval_quality(self) -> Dict[str, Any]:
        """Testa qualidade da recuperação comparando contextos"""
        logger.info("Testando qualidade de recuperação...")

        test_queries = [
            "Bola com velocidade 10 e atrito 2",
            "Resolva para x quando 5x + 7 = 32",
            "A soma de 15 e 20 multiplicada por 3"
        ]

        quality_scores = {
            'genlang': [],
            'traditional_rag': []
        }

        for query in test_queries:
            # Genlang: recupera clusters especializados
            genlang_context = self._get_genlang_context(query)

            # Traditional: recupera chunks
            try:
                if self.traditional_retriever:
                    trad_docs = self.traditional_retriever.invoke(query)
                    trad_context = [doc.page_content for doc in trad_docs]
                else:
                    trad_context = []
            except:
                trad_context = []

            # Calcula relevância (simplificado: conta palavras-chave em comum)
            query_words = set(query.lower().split())

            genlang_score = self._calculate_context_relevance(genlang_context, query_words)
            trad_score = self._calculate_context_relevance(trad_context, query_words)

            quality_scores['genlang'].append(genlang_score)
            quality_scores['traditional_rag'].append(trad_score)

        return quality_scores

    def _calculate_context_relevance(self, contexts: List[str], query_words: set) -> float:
        """Calcula relevância do contexto baseado em overlap de palavras"""
        if not contexts:
            return 0.0

        total_score = 0
        for context in contexts:
            context_words = set(context.lower().split())
            overlap = len(query_words.intersection(context_words))
            total_score += overlap / len(query_words) if query_words else 0
        return total_score / len(contexts)

    def _get_genlang_context(self, query: str) -> List[str]:
        """Recupera contexto usando clusters do Genlang"""
        try:
            query_vector = np.array(self.embeddings.embed_query(query))

            # Encontra clusters mais relevantes
            scores = {}
            for cid, centroid in self.genlang_centroids.items():
                similarity = np.dot(query_vector, centroid) / (np.linalg.norm(query_vector) * np.linalg.norm(centroid))
                scores[cid] = similarity

            # Top 3 clusters
            top_clusters = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]

            contexts = []
            for cid, _ in top_clusters:
                cluster_vectors = self.adaptive_state['memory'].concept_clusters[cid]
                contexts.extend([v.source_text for v in cluster_vectors[:2]])

            return contexts
        except Exception as e:
            logger.error(f"Erro ao recuperar contexto Genlang: {e}")
            return []


# ===============================================
# FUNÇÃO DE EXECUÇÃO DO BENCHMARK
# ===============================================

def run_comprehensive_benchmark(checkpoint_path: str):
    """Executa bateria completa de testes apropriados para o Genlang"""

    try:
        benchmark = DomainFocusedBenchmark(checkpoint_path)
    except Exception as e:
        logger.error(f"Erro ao inicializar benchmark: {e}")
        return

    # 1. Análise da distribuição de treinamento
    print("\n" + "=" * 80)
    print("📊 ANÁLISE DO TREINAMENTO")
    print("=" * 80)
    training_dist = benchmark.test_generator.analyze_training_distribution()
    for task_type, stats in training_dist.items():
        print(f"{task_type}: {stats['count']} exemplos, {stats['success_rate']:.1%} sucesso")

    # 2. Teste de problemas in-distribution
    print("\n" + "=" * 80)
    print("🎯 TESTE 1: PROBLEMAS IN-DISTRIBUTION")
    print("=" * 80)

    in_dist_tests = benchmark.test_generator.generate_in_distribution_tests(num_per_type=5)
    results_in_dist = run_test_suite(benchmark, in_dist_tests)

    # 3. Teste de variações
    print("\n" + "=" * 80)
    print("🔄 TESTE 2: VARIAÇÕES E ADAPTAÇÃO")
    print("=" * 80)

    variation_tests = benchmark.test_generator.generate_slight_variations()
    results_variations = run_test_suite(benchmark, variation_tests)

    # 4. Teste de consistência
    print("\n" + "=" * 80)
    print("📈 TESTE 3: CONSISTÊNCIA")
    print("=" * 80)

    consistency_results = benchmark.test_consistency(num_runs=10)
    print("\nDesvio padrão das respostas (menor é melhor):")
    for system, errors in consistency_results.items():
        if errors:
            print(f"{system}: σ = {np.std(errors):.4f}")

    # 5. Teste de eficiência em lote
    print("\n" + "=" * 80)
    print("⚡ TESTE 4: EFICIÊNCIA EM LOTE")
    print("=" * 80)

    batch_times = benchmark.test_batch_efficiency(num_problems=20)
    print("\nTempo total para 20 problemas:")
    for system, total_time in batch_times.items():
        print(f"{system}: {total_time:.2f}s ({total_time / 20:.3f}s por problema)")

    # 6. Teste de qualidade de recuperação
    print("\n" + "=" * 80)
    print("🔍 TESTE 5: QUALIDADE DE RECUPERAÇÃO")
    print("=" * 80)

    retrieval_quality = benchmark.test_retrieval_quality()
    print("\nScore médio de relevância do contexto:")
    for system, scores in retrieval_quality.items():
        if scores:
            print(f"{system}: {np.mean(scores):.2f}")

    # Visualização final
    create_comprehensive_plots(results_in_dist, results_variations,
                               consistency_results, batch_times, retrieval_quality)


def run_test_suite(benchmark, tests):
    """Executa uma suíte de testes e retorna resultados"""
    results = []

    for test in tests:
        logger.info(f"Executando: {test['problem'][:50]}...")

        for system_name, solver in [
            ('genlang', benchmark.solve_with_genlang),
            ('traditional_rag', benchmark.solve_with_traditional_rag),
            ('pure_llm', benchmark.solve_with_pure_llm)
        ]:
            try:
                result = solver(test['problem'])
                is_correct = (result['solution'] is not None and
                              abs(result['solution'] - test['expected']) < 0.1)

                results.append({
                    'system': system_name,
                    'type': test['type'],
                    'category': test['category'],
                    'correct': is_correct,
                    'time': result.get('time', 0)
                })
            except Exception as e:
                logger.error(f"Erro no sistema {system_name}: {e}")
                results.append({
                    'system': system_name,
                    'type': test['type'],
                    'category': test['category'],
                    'correct': False,
                    'time': 0
                })

    df = pd.DataFrame(results)

    # Resumo por sistema
    if not df.empty:
        summary = df.groupby('system')['correct'].mean()
        print("\nPrecisão por sistema:")
        for system, accuracy in summary.items():
            print(f"{system}: {accuracy:.1%}")

    return df


def create_comprehensive_plots(in_dist_results, variation_results,
                               consistency_results, batch_times, retrieval_quality):
    """Cria visualizações abrangentes dos resultados"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Benchmark Abrangente: Genlang no Seu Domínio', fontsize=16)

    try:
        # 1. Precisão in-distribution
        ax1 = axes[0, 0]
        if not in_dist_results.empty:
            in_dist_summary = in_dist_results.groupby('system')['correct'].mean()
            ax1.bar(in_dist_summary.index, in_dist_summary.values)
        ax1.set_title('Precisão em Problemas do Domínio')
        ax1.set_ylabel('Taxa de Acerto')
        ax1.set_ylim(0, 1.1)

        # 2. Adaptação a variações
        ax2 = axes[0, 1]
        if not variation_results.empty:
            var_summary = variation_results.groupby('system')['correct'].mean()
            ax2.bar(var_summary.index, var_summary.values)
        ax2.set_title('Adaptação a Variações')
        ax2.set_ylabel('Taxa de Acerto')
        ax2.set_ylim(0, 1.1)

        # 3. Consistência
        ax3 = axes[0, 2]
        consistency_stds = {sys: np.std(errs) if errs else 0
                            for sys, errs in consistency_results.items()}
        if consistency_stds:
            ax3.bar(consistency_stds.keys(), consistency_stds.values())
        ax3.set_title('Consistência (menor é melhor)')
        ax3.set_ylabel('Desvio Padrão')

        # 4. Eficiência em lote
        ax4 = axes[1, 0]
        if batch_times:
            ax4.bar(batch_times.keys(), batch_times.values())
        ax4.set_title('Tempo Total (20 problemas)')
        ax4.set_ylabel('Tempo (s)')

        # 5. Tempo por problema
        ax5 = axes[1, 1]
        if batch_times:
            per_problem_times = {k: v / 20 for k, v in batch_times.items()}
            ax5.bar(per_problem_times.keys(), per_problem_times.values())
        ax5.set_title('Tempo Médio por Problema')
        ax5.set_ylabel('Tempo (s)')

        # 6. Qualidade de recuperação
        ax6 = axes[1, 2]
        if retrieval_quality:
            retrieval_means = {sys: np.mean(scores) if scores else 0
                               for sys, scores in retrieval_quality.items()}
            if retrieval_means:
                ax6.bar(retrieval_means.keys(), retrieval_means.values())
        ax6.set_title('Qualidade do Contexto Recuperado')
        ax6.set_ylabel('Score de Relevância')

        plt.tight_layout()
        plt.savefig('genlang_comprehensive_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()

        logger.info("Gráficos salvos como 'genlang_comprehensive_benchmark.png'")

    except Exception as e:
        logger.error(f"Erro ao criar gráficos: {e}")
        print(f"⚠️ Não foi possível gerar os gráficos: {e}")


# ===============================================
# MAIN
# ===============================================

if __name__ == "__main__":
    CHECKPOINT_FILE = "genlang_state_epoch_240.pkl"

    if not os.path.exists(CHECKPOINT_FILE):
        logger.error(f"Arquivo {CHECKPOINT_FILE} não encontrado!")
        print(f"❌ Arquivo {CHECKPOINT_FILE} não encontrado!")
        print("💡 Certifique-se de que o arquivo está no diretório atual")
        sys.exit(1)

    try:
        run_comprehensive_benchmark(CHECKPOINT_FILE)
        logger.info("Benchmark concluído com sucesso!")
    except Exception as e:
        logger.error(f"Erro no benchmark: {e}", exc_info=True)
        print(f"❌ Erro crítico no benchmark: {e}")
        print("🔧 Verifique os logs para mais detalhes")