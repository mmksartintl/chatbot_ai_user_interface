from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="""
Nossa 2ª edição do Treinamento Pricing, Risco e Governança na Comercialização será realizada de forma híbrida, nos dias 03/04 (aula inaugural online), 09 e 10/04 (aulas presenciais em São Paulo/SP no Mercure São Paulo Jardins Hotel).
    """,
)

document_2 = Document(
    page_content="""
O objetivo do curso é oferecer uma vivência abrangente e realista do que significa a gestão de risco na comercialização de energia e como ela se encaixa e influencia os processos de decisão, comunicação e controle ao mesmo tempo que discutiremos com profundidade as diretrizes necessárias para implementar, aprimorar e operacionalizar uma infraestrutura de risco moderna e alinhada com as melhores práticas de mercado
    """,
)

document_3 = Document(
    page_content="""
A partir dos processos de Alocação e Preservação de capital discutiremos desde o impacto dos fatores de risco no resultado financeiro das empresas do setor até a estruturação de processos de gestão de risco, que abrange não só a precificação de operações e avaliação de portfolios, quanto o tratamento e validação da informação, e as estratégias de reporting e comunicação do risco as boas práticas de controle e a interação entre áreas e processos da empresa, tudo isso de uma maneira muito ampla, prática, mas com consistência metodológica.
    """,
)

document_4 = Document(
    page_content="""
Como consequência, discutimos os negócios de comercialização e suas tendências explorando através de casos práticos os principais conceitos associados com a teoria de precificação de operações de energia elétrica e gestão de riscos que passa por uma visão geral de derivativos em energia incluindo os financeiros, processos comerciais como RFQ e RFP, Boletagem e formalização comercial e como se encaixam dentro dessa infraestrutura de gestão de risco, analisando as diretrizes necessárias para sua gestão.
    """,
)

document_5 = Document(
    page_content="""
Com enfoque voltado para aplicações reais típicas do mercado e fazendo uso de recursos didáticos dos mais diversos e interativos o curso também conta com as plataformas quantitativas da Dcide para dar fluidez e racionalidade aos processos, ilustrar conceitos e metodologias complexas associadas com cálculo da volatilidade, estatística avançada, precificação, inteligência de mercado e cálculo e gestão de risco complementando assim os casos de estudos mais simples que podem ser executados em Excel.
    """,
)

document_6 = Document(
    page_content="""
O participante compreenderá a dinâmica dos preços subjacentes aos contratos de energia entendendo seu papel nos processos de inteligência de mercado e precificação de operações, assim como o estudo de suas propriedades e evolução histórica de maneira a entender os diferentes produtos que se negociam no mercado e suas implicações no planejamento energético e gestão de portfólio.
    """,
)

document_7 = Document(
    page_content="""
Ao explicar as principais etapas para colocar em prática os conceitos de precificação dentro dos macroprocessos de alocação de capital em infraestrutura de gestão de risco, discutiremos as etapas necessárias para a correta compreensão e interpretação dos preços Forward e o ferramental estatístico necessário para a sua modelagem, incluindo a estimação da volatilidade e correlações, com todos os tratamentos requeridos, que formam o principal conjunto de entradas para a execução dos modelos de risco.
    """,
)

document_8 = Document(
    page_content="""
O profissional aprenderá como calibrar a volatilidade dos preços para refletir diferentes cenários de liquidez de mercado, escolher corretamente as distribuições de probabilidade dos retornos e calcular intervalos probabilísticos de incerteza, validando as escolhas através de back tests.
    """,
)

document_9 = Document(
    page_content="""
Para sedimentar a discussão teórica sobre as metodologias de precificação e avaliação de risco para produtos de energia, os participantes poderão aplicar as técnicas discutidas através de calculadoras analíticas de propriedade da Dcide para calcular o valor de mercado de operações de energia e precificar contratos mais exóticos como Time Swaps  e opções, além de outros tipos de derivativos e operações estruturadas, estimando o MtM da operação (Mark-to-Market), o desembolso financeiro, a margem de preços, medindo o risco associado com posições líquidas de energia e avaliando o efeito em termos de risco e retorno de incluir determinada operação dentro de um portfólio.
    """,
)

document_10 = Document(
    page_content="""
Uma detalhada discussão sobre métricas de risco e sua interpretação também será feita com foco principalmente nos indicadores V@R (Valor em Risco), CV@R (Valor em Risco Condicional), que são as métricas de risco mais utilizadas pelas empresas do setor, principalmente para se medir o risco marginal que nos permite avaliar a relação risco e retorno de contratos de energia.
    """,
)

document_11 = Document(
    page_content="""
Contaremos também com Risk Denergia para operacionalizar diversas situações do dia-a-dia das empresas mas que requerem sofisticação analítica e de processos para avaliar e estudar diferentes estratégias de alocação de risco e demonstrar de maneira muito prática e consistente os desafios e as habilidades que precisam ser desenvolvidas e dominadas por analistas e áreas de gestão de risco.
    """,
)

document_12 = Document(
    page_content="""
Diversas métricas de risco para várias formas de agregação temporais, quantis da distribuição dos resultados financeiros, variações com respeito à marcação ao mercado, contribuição de produtos para o risco entre outras informações serão exploradas nos exercícios que serão feitos.
    """,
)

document_13 = Document(
    page_content="""
A gestão de risco de crédito e as diretrizes para construção de modelos de scoring no setor de energia serão tratadas, assim como as etapas necessárias para a mensuração de risco de crédito como uma etapa essencial de precificação de contratos e aprovação de operações. As boas práticas em gestão de risco de crédito seguirão os paralelos encontrados no mercado financeiro e documentado nos acordos de basileia.
    """,
)

documents_treinamento = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
    document_11,
    document_12,
]
uuids = [str(uuid4()) for _ in range(len(documents_treinamento))]

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings()

vectordb = FAISS.from_documents(documents=documents_treinamento, embedding=embeddings)
vectordb.save_local("faiss_indx_db")
