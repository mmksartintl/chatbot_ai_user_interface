
from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="""
O que é o FMB Denergia?
Inserida no ecossistema Denergia, o FMB Denergia é uma plataforma integrada, desenvolvida para atender as demandas das áreas de Front, Middle e Back Offices, cobrindo o fluxo completo do mercado de comercialização de energia.

Aderente às regras setoriais, o FMB Denergia foi desenhado para atender as melhores práticas de gestão de contratos e carteiras de energia, cálculo e acompanhamento quantitativo de risco, integração entre processos, sistemas e áreas.

O sistema busca viabilizar então os processos da comercialização de energia partindo de três esferas: os processos principais, baseados na regulação e nas melhores práticas do mercado, comuns a todas as empresas do setor; as atividades específicas em função de ramo de atuação da empresa (comercializadora, geradora, entre outros) e as características particulares de cada cliente, considerando suas formas de reporting e sistemas já existentes.
    """,
)

document_2 = Document(
    page_content="""
Ciclo de vida na gestão de um contrato
O FMB Denergia é uma plataforma estruturada para percorrer os diversos processos de comercialização de energia e permitir aos usuários a gestão eficiente de seus contratos e a redução do risco operacional, organizando cada passo de atividades como o faturamento, a gestão de ativos, entre outros.

Com isso, o FMB, em sua orientação a processos, foi concebido segundo os processos críticos ilustrados abaixo:

A partir disso, o FMB fornece ferramentas de acompanhamento que vão desde o planejamento, permitindo rápidas análises de portfólios customizáveis, passando pelos processos de boletagem de contratos, faturamento e recebimentos até o registro na CCEE e eventuais rentes da emissão da nota correspondente.

Considerando o processo mencionado, o contrato passa a ter papel essencial por ser o elemento principal que direciona os demais ciclos da ferramenta. Os diagramas abaixo são apresentados como forma de resumir o ciclo de vida de um contrato, tendo sua origem na aprovação de uma proposta comercial, seguido pelas etapas realizadas dentro de cada mês de suprimento.

Esse diagrama com suas respectivas etapas pode ser agrupado através de processos, os quais são refletidos nas telas e funcionalidades do FMB. Por exemplo, o processo de exercício das flexilbilidades, está atrelado no software às ferramentas disponíveis na aba “Gestão de Flexibilidades”, incluindo os ciclos de sazonalidade e de Take-or-pay.
    """,
)

document_3 = Document(
    page_content="""
Organização da Informação no FMB
No FMB Denergia a informação é organizada de acordo com a seguinte estrutura:

Entradas: Correspondem aos metadados necessários para execução dos processos, sendo cadastrados pelos usuários ou via integrações.
Agendas: As agendas são os objetos do sistema que irão organizar as informações inseridas e alimentar os ciclos mensais. Um exemplo é a agenda de faturamento que apresenta de forma mensal os montantes de energia, preços negociados e o valor total de cada fatura.
Processos: Os processos são organizados em ciclos mensais, a partir dos quais os usuários conseguem executar as ações necessárias, de exercer uma flexibilidade a emitir uma pré-fatura. Para facilitar a utilização da ferramenta, o usuário tem a opção de uma série de customizações, incluindo a integração com outros softwares, como ERPs de mercado e a Plataforma de Integração da CCEE. Para conhecer melhor os processos que o FMB inclui, clique aqui.
Relatórios: Após a execução dos processos mensais, são disponibilizados para os usuários uma série de relatórios para análises específicas, como o balanço resultante e o resultado comercial.
O diagrama abaixo serve como ilustração dessa organização e elenca exemplos de componentes pertencentes a cada grupo explicado anteriormente:

É importante destacar que no FMB existe uma hierarquia entre as informações, decorrente da própria lógica dos processos, e isso deve ser respeitado durante o cadastro de entidades no sistema. Por exemplo, um perfil de agente precisa estar vinculado a um agente, e o agente, por sua vez, à empresa correspondente. Assim, para facilitar a compreensão dessas relações, desenhamos o seguinte diagrama:
    """,
)

document_4 = Document(
    page_content="""
    Processos da Comercialização
Como uma plataforma orientada a processos, construída com base na experiência de diversos especialistas de todas as áreas do setor, o FMB Denergia ajuda a reduzir substancialmente o risco operacional das empresas, dando visibilidade à camada executiva em todas as etapas do processo, organizando cada passo do faturamento, gestão CCEE, gestão de ativos, contas a pagar e receber, dentre outros.

Os relatórios e interfaces analíticas permitem a cada nível de gestão ter controle sobre o andamento dos processos e seus números, levando a escala executiva um grande repertório de indicadores, potencializando a tomada de decisão e a visão estratégica.
    """,
)

document_5 = Document(
    page_content="""
Gestão de Contratos
O FMB Denergia permite cadastrar e mapear virtualmente todos os contratos do ACL e os principais templates do ACR.

Cobrindo todas as etapas do ciclo de vida destes contratos, o sistema garante uma gestão eficiente, baseada nas melhores práticas do mercado.

Dentre as ferramentas de compliance e controle, destacam-se as de aprovação por e-mail e via sistema, e a gestão de alçadas, que trazem segurança e visibilidade na execução dos processos contratuais.
    """,
)

document_6 = Document(
    page_content="""
Integrações
O FMB Denergia é construído inteiramente sob a base de API’s (Application Programming Interface) disponibilizadas via Web Services, o que permite a extração ou envio de qualquer tipo de informação de maneira direta e simples. Isso faz com que o FMB Denergia seja integrável com qualquer sistema, incluindo aqueles de gestão de acesso como o ActiveDirectory, os principais ERP’s de mercado, os serviços de integração disponíveis na CCEE para gestão do registro e da medição, e principalmente as ferramentas de informação e análise de risco da Dcide, obtendo informações de preços assim que são disponibilizadas e permitindo a medição quantitativa do risco de mercado dos portfólios de energia disponíveis no FMB.
    """,
)

document_7 = Document(
    page_content="""
    Análise de Portfólio
Com um know-how reconhecido no mercado construímos um completo arcabouço para análise e gestão de indicadores de portfólios, incluindo ativos de consumo e geração que passam pelo balanço energético, resultado financeiro, análise de margens e preços médios, médias móveis das carteiras, entre outras.

Somado a relatórios exportáveis para o Excel, existem uma ampla gama de gráficos e funcionalidades de organização e consulta de informação, desenhadas para apresentar a informação de forma eficiente e agradável, alinhada com as principais práticas do setor.

Relatórios customizados adicionais podem ser facilmente montados a partir das informações disponíveis, ou integrados com ferramentas de Data Analytics e Business Inteligence da empresa.
    """,
)

document_8 = Document(
    page_content="""
    Processos CCEE
Considerando todas as particularidades das regras de comercialização e conversando na mesma linguagem do CliqCCEE, os processos CCEE do FMB Denergia foram construídos para mitigar o risco de perdas operacionais e minimizar o impacto da elevada demanda de trabalho no período de registro de contratos. Permitindo controle minucioso em cada etapa do processo, incluindo a verificação das condições de garantias, a gestão de registros contra pagamento e a validação de agentes representados. Tudo isso integrado com a plataforma de serviços da CCEE, o que permite o envio automático dos XML e a verificação dos contratos CCEE.
    """,
)

document_9 = Document(
    page_content="""
    FMB é uma plataforma de gestão de energia orientada a processos e alinhada com as melhores práticas de mercado, construída nativamente para o setor elétrico brasileiro para atender toda a cadeia de etapas relacionadas à comercialização de eletricidade.

Fluída, amigável, robusta e riquíssima em ferramentas analíticas, com interfaces construídas para considerar as especificidades da comercialização de energia, atende as mais rigorosas demandas de compliance e controle, além de ser completamente integrável com os ERP’s de mercado, o CliqCCEE, o SCDE ou ferramentas internas das empresas.
    """,
)

document_10 = Document(
    page_content="""
    Solicitando aprovações
O FMB Denergia, por ser um sistema pautado em conceitos e práticas de compliance, permite que todas as ações do sistema sejam facilmente rastreáveis, no sentido de permitir a identificação de qual usuário realizou certa ação. Além disso, conforme artigos do tópico “Usuários”, cada usuário é configurado com permissões e alçadas correspondentes ao seu escopo de atuação, refletindo nos acessos e possibilidades de ações dentro do FMB.

Nesse sentido, o mecanismo de aprovações é comum a diversos processos do software, funcionando em dois níveis:

Validações: conduzidas pelo próprio usuário operador, para validar valores e entradas realizadas no software.

Aprovações: envio dos objetos (contratos, faturas, entre outros), geralmente ao final dos processos, para a aprovação de usuário com alçada superior correspondente.

Assim, pode-se dizer que as aprovações funcionam no FMB Denergia como um “carimbo” concedido por gerente, diretor ou outro usuário configurado com alçada equivalente, autenticando o objeto ao final de um processo para que este possa caminhar para os próximos ciclos no sistema. Por exemplo, ao final do cadastro de um contrato, geradas suas agendas, este deve ser enviado para aprovação para que possa prosseguir para os ciclos de faturamento, garantias, flexibilidades e outros.

A solicitação de aprovações é realizada de forma similar em todos os processos em que são necessárias, sendo necessário selecionar o objeto que deseja-se aprovar, indicar o nome do usuário aprovador e clicar no botão em destaque “Enviar para aprovação”.
    """,
)

document_11 = Document(
    page_content="""
Cadastrando um novo portfólio
Uma das ferramentas mais importantes de análise disponíveis no FMB Denergia demanda a construção de portfólios. O portfólio funciona como uma “pasta/arquivo” que irá agregar todas as informações dos contratos e parcelas de ativo a ele pertencentes. Essa organização permite o uso da tela “Análise de Portfólio” para estudar, dentre outros aspectos, o balanço de energia e o resultado comercial provenientes do portfólio e suas eventuais exposições."
"Utilizando a Análise de Portfólio
Buscando facilitar a análise e o entendimento da exposição e do resultado financeiro correspondente à carteira do cliente, considerando tanto contratos de compra e venda, como parcelas de ativo, o FMB disponibiliza a tela “Análise de Portfólio”, dentro da aba “Análise” do menu lateral do FMB.

As análises disponíveis nessa tela e suas correspondentes descrições são:

Balanço por fonte: agregados através das fontes Convencional e Incentivada (0%, 50%, 80% e 100%) são apresentados os montantes de energia de requisito, composto por vendas, recurso, correspondente aos contratos de compra e garantia física/geração, além do balanço resultante.
Balanço por submercado: apresenta os montantes de energia e o balanço/exposição correspondente através de uma agregação pelos submercados do SIN (SE/CO, S, NE e N).
Resultado financeiro: visão financeira do portfólio, apresentando o resultado comercial proveniente dos contratos de compra e venda além do efeito da liquidação da posição em aberto, podendo considerar o PLD e uma curva de preços Forward.
Média móvel por fonte:  cálculo da média móvel dos montantes de energia, sendo aplicado para gestão e lastro.
Preço médio: relatório de preços médios, calculados para receita (vendas + sobras) e despesa (compras + déficits), além da margem de comercialização resultante.
Relatório FIFO [disponível apenas através de customização]: Relatório de preços considerando a metodologia FIFO (First In, First Out).
Para utilizar as funcionalidades dessa tela, basta escolher o tipo de análise, selecionar o portfólio e os correspondentes parâmetros (unidade, período, curva Forward, valores utilizados e agregação temporal), clicando, ao final, em “Executa análise”.
    """,
)

document_12 = Document(
    page_content="""
Como está organizado o faturamento?
No FMB Denergia o processo de faturamento é dividido em dois ciclos: o ciclo de faturamento (vendas/compras) e a emissão de notas fiscais.

No ciclo de faturamento, temos as etapas de validação da energia, validação do preço e emissão da pré-fatura.

Energia – Nessa etapa é necessário o exercício das flexibilidades para definição das quantidades finais a serem faturadas, e posteriormente é feita a validação desses valores.
Preço – Nas atividades correspondente ao preço das faturas estão disponíveis a validação do reajuste, de preços variáveis e regras de preços, como caps, floors e collars. Ao final do processamento dessas condições, é também necessária a validação dos preços finais.
Pré-fatura – Após a conclusão as etapas de energia e preço, a caixa de pré-fatura estará disponível para executar."
"Cadastro de Contratos CCEE
O FMB Denergia, sendo uma solução completa para front, middle e back-offices de empresas do setor elétrico, passa também pelos processos de registro na CCEE. Esse registro é dividido em quatro processos principais: a sincronização de contratos CCEE (registro de contratos), o ciclo mensal de ajustes de vendas, a validação de registros de compras e o ciclo mensal de validação de ajustes de compras.
    """,
)

document_13 = Document(
    page_content="""
Workflow de propostas – Da proposta ao contrato
O FMB Denergia disponibiliza, dentre suas funcionalidades, a criação e o acompanhamento de propostas. O menu de propostas do FMB permite então que todo o processo comercial seja mapeado pelos usuários, passando pelas etapas de confecção, aprovação, envio e aceite, com a conversão, ao final, das propostas aceitas em contratos FMB para o faturamento e demais ciclos. Assim, a utilização dessa ferramenta confere agilidade ao processo de cadastro de operações através de uma tela simples, onde já serão indicadas as informações essenciais do futuro contrato, como contraparte, energia, preço e período de suprimento.

Com isso, após a criação de uma proposta (leia mais aqui), o usuário poderá caminhar por um workflow específico, detalhado na imagem abaixo. Os status de proposta são, então:

Em confecção: etapa inicial após o preenchimento das informações de criação;
Aguardando aprovação: envio para aprovação de usuário com permissão adequada (gestor/gerente/responsável);
Aprovada: status após aprovação, podendo ser enviada para a contraparte ou cancelada;
Enviada: registro de que houve envio para a contraparte, ficando pendente a resposta, positiva ou negativa, do aceite;
Aceita: proposta aceita pela contraparte, pronta para ser convertida em contrato FMB para que se dê início aos processos de suprimento, faturamento e registro;
Convertida: Proposta convertida em contrato FMB;
    """,
)

documents = [
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
uuids = [str(uuid4()) for _ in range(len(documents))]

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings()

vectordb = FAISS.from_documents(documents=documents, embedding=embeddings)
vectordb.save_local("faiss_indx_db")
