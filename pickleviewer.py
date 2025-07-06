import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime
import warnings
from collections import deque
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')


# ===============================================
# CLASSES NECESSÁRIAS PARA DESSERIALIZAR O PKL
# ===============================================

# Recriar as classes do sistema original para permitir o unpickling
@dataclass
class GenlangVector:
    vector: np.ndarray
    source_text: str
    source_agent: str

    def similarity(self, other: 'GenlangVector') -> float:
        cos_sim = np.dot(self.vector, other.vector) / (np.linalg.norm(self.vector) * np.linalg.norm(other.vector))
        return float(cos_sim)


class CognitiveState(Enum):
    RIGID = "rigid"
    OPTIMAL = "optimal"
    CHAOTIC = "chaotic"


class MemorySystem:
    def __init__(self, cluster_threshold: float = 0.8):
        self.concept_clusters: Dict[str, List[GenlangVector]] = {}
        self.cluster_threshold = cluster_threshold


class CriticalityGovernor:
    def __init__(self, history_size: int = 100):
        self.communication_history = deque(maxlen=history_size)
        self.task_success_history = deque(maxlen=history_size)


# ===============================================
# VISUALIZADOR PRINCIPAL
# ===============================================

class GenlangPKLViewer:
    """Visualizador para arquivos PKL do sistema Genlang"""

    def __init__(self, pkl_file_path: str):
        self.pkl_file_path = pkl_file_path
        self.state_data = None
        self.loaded = False

    def load_pkl_file(self) -> bool:
        """Carrega o arquivo PKL com tratamento de erros de importação"""
        try:
            if not os.path.exists(self.pkl_file_path):
                print(f"❌ Arquivo não encontrado: {self.pkl_file_path}")
                return False

            # Tentar carregar com diferentes estratégias
            try:
                with open(self.pkl_file_path, 'rb') as f:
                    self.state_data = pickle.load(f)
            except (AttributeError, ModuleNotFoundError) as e:
                print(f"⚠️ Erro de importação detectado. Tentando estratégia alternativa...")
                return self._load_with_custom_unpickler()

            self.loaded = True
            print(f"✅ Arquivo carregado com sucesso: {self.pkl_file_path}")
            return True

        except Exception as e:
            print(f"❌ Erro ao carregar arquivo: {e}")
            return False

    def _load_with_custom_unpickler(self) -> bool:
        """Carrega com unpickler customizado para resolver problemas de módulo"""
        try:
            import sys
            import types

            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Mapear classes do módulo original para as classes locais
                    if name == 'MemorySystem':
                        return MemorySystem
                    elif name == 'CriticalityGovernor':
                        return CriticalityGovernor
                    elif name == 'GenlangVector':
                        return GenlangVector
                    elif name == 'CognitiveState':
                        return CognitiveState
                    else:
                        # Tentar carregar normalmente
                        try:
                            return super().find_class(module, name)
                        except:
                            # Se falhar, criar uma classe dummy
                            return type(name, (), {})

            with open(self.pkl_file_path, 'rb') as f:
                unpickler = CustomUnpickler(f)
                self.state_data = unpickler.load()

            self.loaded = True
            print(f"✅ Arquivo carregado com unpickler customizado: {self.pkl_file_path}")
            return True

        except Exception as e:
            print(f"❌ Falha no unpickler customizado: {e}")
            return self._load_as_raw_data()

    def _load_as_raw_data(self) -> bool:
        """Última tentativa: carregar dados brutos ignorando tipos"""
        try:
            # Ler o arquivo como bytes e tentar extrair informações básicas
            with open(self.pkl_file_path, 'rb') as f:
                raw_data = f.read()

            print(f"⚠️ Carregamento como dados brutos. Funcionalidade limitada.")
            print(f"📊 Tamanho do arquivo: {len(raw_data)} bytes")

            # Tentar encontrar strings no arquivo para dar alguma informação
            try:
                decoded = raw_data.decode('latin-1', errors='ignore')
                if 'memory' in decoded.lower():
                    print("🧠 Detectado: Sistema de memória")
                if 'governor' in decoded.lower():
                    print("🏛️ Detectado: Governador")
                if 'task_history' in decoded.lower():
                    print("📋 Detectado: Histórico de tarefas")
            except:
                pass

            self.state_data = {'raw_data': raw_data, 'error': 'Could not unpickle properly'}
            self.loaded = True
            return True

        except Exception as e:
            print(f"❌ Falha completa no carregamento: {e}")
            return False

    def get_file_info(self) -> Dict[str, Any]:
        """Retorna informações básicas sobre o arquivo"""
        if not self.loaded:
            return {}

        info = {
            'file_path': self.pkl_file_path,
            'file_size_mb': os.path.getsize(self.pkl_file_path) / (1024 * 1024),
            'modification_time': datetime.fromtimestamp(os.path.getmtime(self.pkl_file_path)),
        }

        if isinstance(self.state_data, dict) and 'error' not in self.state_data:
            info['keys_in_state'] = list(self.state_data.keys())
            info['status'] = 'Carregado com sucesso'
        else:
            info['keys_in_state'] = 'Não disponível'
            info['status'] = 'Carregamento parcial ou com erros'

        return info

    def explore_structure(self) -> None:
        """Explora a estrutura dos dados salvos"""
        if not self.loaded:
            print("❌ Arquivo não carregado")
            return

        print("🔍 ESTRUTURA DOS DADOS SALVOS")
        print("=" * 50)

        if isinstance(self.state_data, dict) and 'error' not in self.state_data:
            for key, value in self.state_data.items():
                print(f"\n📁 {key}:")
                print(f"   Tipo: {type(value).__name__}")

                try:
                    if hasattr(value, '__dict__'):
                        attrs = [attr for attr in dir(value) if not attr.startswith('_')]
                        print(f"   Atributos públicos: {attrs[:10]}{'...' if len(attrs) > 10 else ''}")
                    elif isinstance(value, (list, deque)):
                        print(f"   Tamanho: {len(value)}")
                        if value:
                            print(f"   Tipo dos elementos: {type(value[0]).__name__}")
                    elif isinstance(value, dict):
                        print(f"   Chaves: {list(value.keys())[:10]}{'...' if len(value) > 10 else ''}")
                        print(f"   Tamanho: {len(value)}")
                except Exception as e:
                    print(f"   Erro ao analisar: {e}")
        else:
            print(f"⚠️ Dados com estrutura não padrão ou erro no carregamento")
            if 'error' in self.state_data:
                print(f"Erro: {self.state_data['error']}")

    def analyze_memory_system(self) -> None:
        """Analisa o sistema de memória"""
        if not self.loaded:
            print("❌ Arquivo não carregado")
            return

        print("\n🧠 ANÁLISE DO SISTEMA DE MEMÓRIA")
        print("=" * 50)

        try:
            if 'memory' not in self.state_data:
                print("❌ Sistema de memória não encontrado")
                return

            memory = self.state_data['memory']

            if hasattr(memory, 'concept_clusters'):
                clusters = memory.concept_clusters
                print(f"📊 Número de clusters conceituais: {len(clusters)}")

                if clusters:
                    # Estatísticas dos clusters
                    cluster_sizes = []
                    valid_clusters = 0

                    for cluster_id, vectors in clusters.items():
                        try:
                            if isinstance(vectors, (list, deque)) and vectors:
                                cluster_sizes.append(len(vectors))
                                valid_clusters += 1
                        except:
                            continue

                    if cluster_sizes:
                        print(f"📈 Clusters válidos: {valid_clusters}")
                        print(f"📊 Tamanho médio dos clusters: {np.mean(cluster_sizes):.2f}")
                        print(f"📉 Tamanho mínimo: {np.min(cluster_sizes)}")
                        print(f"📈 Tamanho máximo: {np.max(cluster_sizes)}")

                        # Top 5 maiores clusters
                        sorted_clusters = sorted(clusters.items(),
                                                 key=lambda x: len(x[1]) if isinstance(x[1], (list, deque)) else 0,
                                                 reverse=True)

                        print("\n🏆 Top 5 maiores clusters:")
                        for i, (cluster_id, vectors) in enumerate(sorted_clusters[:5]):
                            if isinstance(vectors, (list, deque)) and vectors:
                                print(f"   {i + 1}. {cluster_id}: {len(vectors)} vetores")

                                # Tentar mostrar textos de exemplo
                                try:
                                    sample_texts = []
                                    for v in list(vectors)[:3]:
                                        if hasattr(v, 'source_text'):
                                            text = v.source_text
                                            text = text[:50] + "..." if len(text) > 50 else text
                                            sample_texts.append(text)
                                    if sample_texts:
                                        print(f"      Exemplos: {sample_texts}")
                                except Exception as e:
                                    print(f"      Erro ao extrair exemplos: {e}")
                    else:
                        print("⚠️ Nenhum cluster válido encontrado")
                else:
                    print("⚠️ Clusters conceituais vazios")
            else:
                print("⚠️ Atributo concept_clusters não encontrado")

        except Exception as e:
            print(f"❌ Erro na análise da memória: {e}")

    def analyze_governor(self) -> None:
        """Analisa o governador de criticalidade"""
        if not self.loaded:
            print("❌ Arquivo não carregado")
            return

        print("\n🏛️ ANÁLISE DO GOVERNADOR")
        print("=" * 50)

        try:
            if 'governor' not in self.state_data:
                print("❌ Governador não encontrado")
                return

            governor = self.state_data['governor']

            # Análise do histórico de comunicação
            if hasattr(governor, 'communication_history'):
                comm_hist = governor.communication_history
                if isinstance(comm_hist, (list, deque)):
                    print(f"💬 Histórico de comunicação: {len(comm_hist)} trocas")

                    if comm_hist:
                        # Análise dos agentes
                        agent_counts = {}
                        valid_exchanges = 0

                        for exchange in comm_hist:
                            try:
                                if hasattr(exchange, 'source_agent'):
                                    agent = exchange.source_agent
                                    agent_counts[agent] = agent_counts.get(agent, 0) + 1
                                    valid_exchanges += 1
                            except:
                                continue

                        if agent_counts:
                            print(f"🤖 Trocas válidas analisadas: {valid_exchanges}")
                            print("🤖 Participação dos agentes:")
                            for agent, count in sorted(agent_counts.items(), key=lambda x: x[1], reverse=True):
                                percentage = count / valid_exchanges * 100 if valid_exchanges > 0 else 0
                                print(f"   {agent}: {count} contribuições ({percentage:.1f}%)")

            # Análise do histórico de sucesso
            if hasattr(governor, 'task_success_history'):
                success_hist = governor.task_success_history
                if isinstance(success_hist, (list, deque)):
                    print(f"\n📊 Histórico de sucesso: {len(success_hist)} registros")
                    if success_hist:
                        # Converter para números válidos
                        valid_scores = []
                        for score in success_hist:
                            try:
                                if isinstance(score, (int, float)) and not np.isnan(score):
                                    valid_scores.append(float(score))
                            except:
                                continue

                        if valid_scores:
                            print(f"✅ Registros válidos: {len(valid_scores)}")
                            print(f"📈 Taxa de sucesso média: {np.mean(valid_scores):.3f}")
                            print(f"📊 Melhor performance: {np.max(valid_scores):.3f}")
                            print(f"📉 Pior performance: {np.min(valid_scores):.3f}")
                            print(f"📐 Desvio padrão: {np.std(valid_scores):.3f}")

        except Exception as e:
            print(f"❌ Erro na análise do governador: {e}")

    def analyze_task_history(self) -> None:
        """Analisa o histórico de tarefas"""
        if not self.loaded:
            print("❌ Arquivo não carregado")
            return

        print("\n📋 ANÁLISE DO HISTÓRICO DE TAREFAS")
        print("=" * 50)

        try:
            if 'task_history' not in self.state_data:
                print("❌ Histórico de tarefas não encontrado")
                return

            task_history = self.state_data['task_history']

            if not task_history:
                print("⚠️ Histórico de tarefas vazio")
                return

            print(f"📊 Total de registros: {len(task_history)}")

            # Converter para DataFrame para análise
            try:
                df = pd.DataFrame(task_history)
                print(f"📋 Colunas disponíveis: {list(df.columns)}")

                # Análise por tipo de tarefa
                if 'task_type' in df.columns:
                    print("\n🔍 Distribuição por tipo de tarefa:")
                    type_counts = df['task_type'].value_counts()
                    for task_type, count in type_counts.items():
                        if 'success' in df.columns:
                            success_rate = df[df['task_type'] == task_type]['success'].mean()
                            print(f"   {task_type}: {count} tarefas (sucesso: {success_rate:.3f})")
                        else:
                            print(f"   {task_type}: {count} tarefas")

                # Análise temporal
                if 'epoch' in df.columns:
                    epochs = df['epoch']
                    print(f"\n⏰ Épocas analisadas: {epochs.min()} - {epochs.max()}")

                    if 'success' in df.columns:
                        overall_success = df['success'].mean()
                        print(f"📈 Taxa de sucesso geral: {overall_success:.3f}")

                        # Evolução ao longo das épocas
                        success_by_epoch = df.groupby('epoch')['success'].mean()
                        print(f"📊 Melhor época: {success_by_epoch.idxmax()} (sucesso: {success_by_epoch.max():.3f})")
                        print(f"📉 Pior época: {success_by_epoch.idxmin()} (sucesso: {success_by_epoch.min():.3f})")

                    if 'weighted_success' in df.columns:
                        weighted_success = df['weighted_success'].mean()
                        print(f"⚖️ Sucesso ponderado médio: {weighted_success:.3f}")

                # Análise de eficiência
                if 'exchanges' in df.columns:
                    avg_exchanges = df['exchanges'].mean()
                    print(f"\n💬 Número médio de trocas: {avg_exchanges:.2f}")
                    print(f"📊 Mínimo de trocas: {df['exchanges'].min()}")
                    print(f"📈 Máximo de trocas: {df['exchanges'].max()}")

            except Exception as e:
                print(f"❌ Erro na análise com DataFrame: {e}")
                # Análise simples sem pandas
                print("🔄 Tentando análise simplificada...")
                if isinstance(task_history[0], dict):
                    keys = set()
                    for task in task_history:
                        keys.update(task.keys())
                    print(f"🔑 Chaves encontradas: {list(keys)}")

        except Exception as e:
            print(f"❌ Erro na análise do histórico: {e}")

    def plot_learning_evolution(self) -> None:
        """Plota a evolução do aprendizado"""
        if not self.loaded:
            print("❌ Arquivo não carregado")
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Evolução do Sistema Genlang\n{os.path.basename(self.pkl_file_path)}', fontsize=16)

            plots_created = 0

            # 1. Evolução da taxa de sucesso
            if 'task_history' in self.state_data and self.state_data['task_history']:
                try:
                    df_tasks = pd.DataFrame(self.state_data['task_history'])
                    if 'epoch' in df_tasks.columns and 'success' in df_tasks.columns:
                        success_by_epoch = df_tasks.groupby('epoch')['success'].mean()
                        axes[0, 0].plot(success_by_epoch.index, success_by_epoch.values, 'b-o', alpha=0.7)
                        axes[0, 0].set_title('Taxa de Sucesso por Época')
                        axes[0, 0].set_xlabel('Época')
                        axes[0, 0].set_ylabel('Taxa de Sucesso')
                        axes[0, 0].grid(True, alpha=0.3)
                        axes[0, 0].set_ylim(0, 1)
                        plots_created += 1
                except Exception as e:
                    axes[0, 0].text(0.5, 0.5, f'Erro no gráfico\nde sucesso:\n{str(e)[:50]}...',
                                    ha='center', va='center', transform=axes[0, 0].transAxes)
                    axes[0, 0].set_title('Taxa de Sucesso - Erro')

            # 2. Evolução do vocabulário
            if 'centroid_history' in self.state_data and self.state_data['centroid_history']:
                try:
                    centroid_hist = self.state_data['centroid_history']
                    epochs = []
                    vocab_sizes = []

                    for h in centroid_hist:
                        if isinstance(h, dict) and 'epoch' in h and 'centroids' in h:
                            epochs.append(h['epoch'])
                            vocab_sizes.append(len(h['centroids']))

                    if epochs and vocab_sizes:
                        axes[0, 1].plot(epochs, vocab_sizes, 'g-s', alpha=0.7)
                        axes[0, 1].set_title('Crescimento do Vocabulário')
                        axes[0, 1].set_xlabel('Época')
                        axes[0, 1].set_ylabel('Tamanho do Vocabulário')
                        axes[0, 1].grid(True, alpha=0.3)
                        plots_created += 1
                except Exception as e:
                    axes[0, 1].text(0.5, 0.5, f'Erro no gráfico\nde vocabulário:\n{str(e)[:50]}...',
                                    ha='center', va='center', transform=axes[0, 1].transAxes)
                    axes[0, 1].set_title('Vocabulário - Erro')

            # 3. Distribuição de tipos de tarefa
            if 'task_history' in self.state_data and self.state_data['task_history']:
                try:
                    df_tasks = pd.DataFrame(self.state_data['task_history'])
                    if 'task_type' in df_tasks.columns:
                        task_type_counts = df_tasks['task_type'].value_counts()
                        if len(task_type_counts) > 0:
                            axes[1, 0].pie(task_type_counts.values, labels=task_type_counts.index,
                                           autopct='%1.1f%%', startangle=90)
                            axes[1, 0].set_title('Distribuição de Tipos de Tarefa')
                            plots_created += 1
                except Exception as e:
                    axes[1, 0].text(0.5, 0.5, f'Erro no gráfico\nde distribuição:\n{str(e)[:50]}...',
                                    ha='center', va='center', transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title('Distribuição - Erro')

            # 4. Métricas de governança
            if 'metrics_history' in self.state_data and self.state_data['metrics_history']:
                try:
                    df_metrics = pd.DataFrame(self.state_data['metrics_history'])
                    if 'epoch' in df_metrics.columns:
                        metrics_plotted = 0
                        for metric in ['novelty', 'coherence', 'grounding']:
                            if metric in df_metrics.columns:
                                axes[1, 1].plot(df_metrics['epoch'], df_metrics[metric],
                                                label=metric.title(), alpha=0.7)
                                metrics_plotted += 1

                        if metrics_plotted > 0:
                            axes[1, 1].set_title('Métricas de Governança')
                            axes[1, 1].set_xlabel('Época')
                            axes[1, 1].set_ylabel('Valor da Métrica')
                            axes[1, 1].legend()
                            axes[1, 1].grid(True, alpha=0.3)
                            axes[1, 1].set_ylim(0, 1)
                            plots_created += 1
                except Exception as e:
                    axes[1, 1].text(0.5, 0.5, f'Erro no gráfico\nde métricas:\n{str(e)[:50]}...',
                                    ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Métricas - Erro')

            # Se nenhum gráfico foi criado com sucesso
            if plots_created == 0:
                fig.text(0.5, 0.5, 'Nenhum gráfico pôde ser gerado\ncom os dados disponíveis',
                         ha='center', va='center', fontsize=16)

            plt.tight_layout()
            plt.show()

            if plots_created > 0:
                print(f"✅ {plots_created} gráfico(s) gerado(s) com sucesso")
            else:
                print("⚠️ Nenhum gráfico foi gerado devido a problemas nos dados")

        except Exception as e:
            print(f"❌ Erro ao gerar gráficos: {e}")

    def export_summary_report(self, output_file: str = None) -> None:
        """Exporta um relatório resumido"""
        if not self.loaded:
            print("❌ Arquivo não carregado")
            return

        if output_file is None:
            base_name = os.path.splitext(os.path.basename(self.pkl_file_path))[0]
            output_file = f"report_{base_name}.txt"

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("RELATÓRIO DE ANÁLISE DO SISTEMA GENLANG\n")
                f.write("=" * 50 + "\n\n")

                # Informações do arquivo
                info = self.get_file_info()
                f.write(f"Arquivo: {info['file_path']}\n")
                f.write(f"Tamanho: {info['file_size_mb']:.2f} MB\n")
                f.write(f"Modificado em: {info['modification_time']}\n")
                f.write(f"Status: {info['status']}\n")
                f.write(f"Chaves principais: {info['keys_in_state']}\n\n")

                # Análise da memória
                if 'memory' in self.state_data:
                    try:
                        memory = self.state_data['memory']
                        if hasattr(memory, 'concept_clusters'):
                            clusters = memory.concept_clusters
                            f.write(f"MEMÓRIA:\n")
                            f.write(f"- Clusters conceituais: {len(clusters)}\n")
                            if clusters:
                                valid_clusters = [v for v in clusters.values()
                                                  if isinstance(v, (list, deque)) and v]
                                if valid_clusters:
                                    cluster_sizes = [len(v) for v in valid_clusters]
                                    f.write(f"- Clusters válidos: {len(valid_clusters)}\n")
                                    f.write(f"- Tamanho médio dos clusters: {np.mean(cluster_sizes):.2f}\n")
                                    f.write(f"- Maior cluster: {np.max(cluster_sizes)} vetores\n\n")
                    except Exception as e:
                        f.write(f"MEMÓRIA: Erro na análise - {e}\n\n")

                # Análise das tarefas
                if 'task_history' in self.state_data and self.state_data['task_history']:
                    try:
                        df = pd.DataFrame(self.state_data['task_history'])
                        f.write(f"HISTÓRICO DE TAREFAS:\n")
                        f.write(f"- Total de tarefas: {len(df)}\n")

                        if 'success' in df.columns:
                            f.write(f"- Taxa de sucesso geral: {df['success'].mean():.3f}\n")

                        if 'task_type' in df.columns:
                            f.write(f"- Tipos de tarefa: {df['task_type'].nunique()}\n")
                            for task_type in df['task_type'].unique():
                                subset = df[df['task_type'] == task_type]
                                if 'success' in df.columns:
                                    success_rate = subset['success'].mean()
                                    f.write(f"  * {task_type}: {len(subset)} tarefas, sucesso: {success_rate:.3f}\n")
                                else:
                                    f.write(f"  * {task_type}: {len(subset)} tarefas\n")
                    except Exception as e:
                        f.write(f"HISTÓRICO DE TAREFAS: Erro na análise - {e}\n")

            print(f"✅ Relatório exportado para: {output_file}")

        except Exception as e:
            print(f"❌ Erro ao exportar relatório: {e}")

    def interactive_explorer(self) -> None:
        """Interface interativa para explorar os dados"""
        if not self.loaded:
            print("❌ Arquivo não carregado")
            return

        while True:
            print("\n" + "=" * 50)
            print("🔍 EXPLORADOR INTERATIVO DE PKL")
            print("=" * 50)
            print("1. Informações do arquivo")
            print("2. Estrutura dos dados")
            print("3. Análise da memória")
            print("4. Análise do governador")
            print("5. Histórico de tarefas")
            print("6. Gráficos de evolução")
            print("7. Exportar relatório")
            print("8. Sair")

            try:
                choice = input("\nEscolha uma opção (1-8): ").strip()

                if choice == '1':
                    info = self.get_file_info()
                    print(f"\n📁 Arquivo: {info['file_path']}")
                    print(f"📊 Tamanho: {info['file_size_mb']:.2f} MB")
                    print(f"⏰ Modificado: {info['modification_time']}")
                    print(f"📊 Status: {info['status']}")
                    print(f"🔑 Chaves: {info['keys_in_state']}")

                elif choice == '2':
                    self.explore_structure()

                elif choice == '3':
                    self.analyze_memory_system()

                elif choice == '4':
                    self.analyze_governor()

                elif choice == '5':
                    self.analyze_task_history()

                elif choice == '6':
                    self.plot_learning_evolution()

                elif choice == '7':
                    output_file = input("Nome do arquivo de relatório (Enter para padrão): ").strip()
                    if not output_file:
                        output_file = None
                    self.export_summary_report(output_file)

                elif choice == '8':
                    print("👋 Até logo!")
                    break

                else:
                    print("❌ Opção inválida")

            except KeyboardInterrupt:
                print("\n👋 Interrompido pelo usuário")
                break
            except Exception as e:
                print(f"❌ Erro: {e}")

            input("\nPressione Enter para continuar...")


# Função principal para uso direto
def visualize_pkl(pkl_file_path: str, interactive: bool = True):
    """Função principal para visualizar arquivos PKL"""
    viewer = GenlangPKLViewer(pkl_file_path)

    if not viewer.load_pkl_file():
        return None

    if interactive:
        viewer.interactive_explorer()
    else:
        # Análise rápida
        print("🔍 ANÁLISE RÁPIDA")
        print("=" * 50)
        viewer.explore_structure()
        viewer.analyze_memory_system()
        viewer.analyze_governor()
        viewer.analyze_task_history()
        viewer.plot_learning_evolution()

    return viewer


def scan_pkl_files(directory: str = ".") -> List[str]:
    """Escaneia diretório em busca de arquivos PKL"""
    pkl_files = []
    try:
        for file in os.listdir(directory):
            if file.endswith('.pkl'):
                full_path = os.path.join(directory, file)
                pkl_files.append(full_path)
    except Exception as e:
        print(f"❌ Erro ao escanear diretório: {e}")

    return sorted(pkl_files)


def compare_pkl_files(file1: str, file2: str) -> None:
    """Compara dois arquivos PKL"""
    print(f"\n🔍 COMPARAÇÃO DE ARQUIVOS PKL")
    print("=" * 50)

    viewer1 = GenlangPKLViewer(file1)
    viewer2 = GenlangPKLViewer(file2)

    if not viewer1.load_pkl_file() or not viewer2.load_pkl_file():
        print("❌ Falha ao carregar um ou ambos os arquivos")
        return

    print(f"📁 Arquivo 1: {os.path.basename(file1)}")
    print(f"📁 Arquivo 2: {os.path.basename(file2)}")

    # Comparar informações básicas
    info1 = viewer1.get_file_info()
    info2 = viewer2.get_file_info()

    print(f"\n📊 COMPARAÇÃO DE TAMANHOS:")
    print(f"   Arquivo 1: {info1['file_size_mb']:.2f} MB")
    print(f"   Arquivo 2: {info2['file_size_mb']:.2f} MB")

    # Comparar históricos se disponíveis
    try:
        if ('task_history' in viewer1.state_data and 'task_history' in viewer2.state_data and
                viewer1.state_data['task_history'] and viewer2.state_data['task_history']):

            df1 = pd.DataFrame(viewer1.state_data['task_history'])
            df2 = pd.DataFrame(viewer2.state_data['task_history'])

            print(f"\n📋 COMPARAÇÃO DE HISTÓRICOS:")
            print(f"   Arquivo 1: {len(df1)} tarefas")
            print(f"   Arquivo 2: {len(df2)} tarefas")

            if 'success' in df1.columns and 'success' in df2.columns:
                success1 = df1['success'].mean()
                success2 = df2['success'].mean()
                print(f"   Taxa de sucesso 1: {success1:.3f}")
                print(f"   Taxa de sucesso 2: {success2:.3f}")
                print(f"   Diferença: {success2 - success1:.3f}")

        # Comparar vocabulário
        if ('memory' in viewer1.state_data and 'memory' in viewer2.state_data):
            memory1 = viewer1.state_data['memory']
            memory2 = viewer2.state_data['memory']

            if (hasattr(memory1, 'concept_clusters') and hasattr(memory2, 'concept_clusters')):
                vocab1 = len(memory1.concept_clusters)
                vocab2 = len(memory2.concept_clusters)
                print(f"\n🧠 COMPARAÇÃO DE VOCABULÁRIO:")
                print(f"   Arquivo 1: {vocab1} clusters")
                print(f"   Arquivo 2: {vocab2} clusters")
                print(f"   Crescimento: {vocab2 - vocab1} clusters")

    except Exception as e:
        print(f"❌ Erro na comparação detalhada: {e}")


# Exemplo de uso principal
if __name__ == "__main__":
    print("🔍 VISUALIZADOR DE ARQUIVOS PKL - SISTEMA GENLANG")
    print("=" * 60)

    # Escanear arquivos PKL disponíveis
    pkl_files = scan_pkl_files()

    if not pkl_files:
        print("❌ Nenhum arquivo PKL encontrado no diretório atual")


        # Tentar arquivo específico se fornecido
        test_file = "genlang_state_epoch_60.pkl"
        if os.path.exists(test_file):
            print(f"🔍 Tentando carregar: {test_file}")
            viewer = visualize_pkl(test_file, interactive=True)
        else:
            print("💡 Coloque seus arquivos .pkl no diretório e execute novamente")
    else:
        print(f"📁 Encontrados {len(pkl_files)} arquivo(s) PKL:")
        for i, f in enumerate(pkl_files, 1):
            file_size = os.path.getsize(f) / (1024 * 1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(f))
            print(f"   {i}. {os.path.basename(f)} ({file_size:.2f} MB, {mod_time.strftime('%Y-%m-%d %H:%M')})")

        # Menu principal
        while True:
            print(f"\n{'=' * 50}")
            print("OPÇÕES:")
            print("1-{}: Analisar arquivo específico".format(len(pkl_files)))
            print("C: Comparar dois arquivos")
            print("A: Analisar todos os arquivos")
            print("Q: Sair")

            choice = input("\nEscolha uma opção: ").strip().upper()

            if choice == 'Q':
                print("👋 Até logo!")
                break
            elif choice == 'C':
                if len(pkl_files) < 2:
                    print("❌ Precisa de pelo menos 2 arquivos para comparar")
                    continue

                print("Escolha o primeiro arquivo:")
                for i, f in enumerate(pkl_files, 1):
                    print(f"   {i}. {os.path.basename(f)}")

                try:
                    idx1 = int(input("Arquivo 1: ")) - 1
                    idx2 = int(input("Arquivo 2: ")) - 1

                    if 0 <= idx1 < len(pkl_files) and 0 <= idx2 < len(pkl_files):
                        compare_pkl_files(pkl_files[idx1], pkl_files[idx2])
                    else:
                        print("❌ Índices inválidos")
                except ValueError:
                    print("❌ Entrada inválida")

            elif choice == 'A':
                print("🔍 Analisando todos os arquivos...")
                for pkl_file in pkl_files:
                    print(f"\n{'=' * 30}")
                    print(f"📁 Analisando: {os.path.basename(pkl_file)}")
                    print(f"{'=' * 30}")
                    viewer = visualize_pkl(pkl_file, interactive=False)

            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(pkl_files):
                    selected_file = pkl_files[idx]
                    print(f"🔍 Analisando: {os.path.basename(selected_file)}")
                    viewer = visualize_pkl(selected_file, interactive=True)
                else:
                    print("❌ Número inválido")
            else:
                print("❌ Opção inválida")