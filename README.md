tcc-fraude/
│
├── data/                  # Dados brutos e processados (não versionar os brutos grandes)
│   ├── raw/                # Dados originais (ex.: train_transaction.csv)
│   ├── interim/            # Dados intermediários após limpeza
│   └── processed/          # Dados prontos para modelagem
│
├── notebooks/             # Jupyter Notebooks para experimentação
│   ├── 01_exploracao.ipynb
│   ├── 02_preprocessamento.ipynb
│   ├── 03_modelagem.ipynb
│   └── 04_avaliacao.ipynb
│
├── src/                   # Código fonte reutilizável (funções e scripts)
│   ├── __init__.py
│   ├── data_prep.py       # Funções de limpeza e transformação
│   ├── features.py        # Feature engineering
│   ├── models.py          # Treinamento e avaliação de modelos
│   └── utils.py           # Funções auxiliares
│
├── reports/               # Documentos e resultados
│   ├── figures/           # Gráficos e imagens para o TCC
│   ├── tabelas/           # Tabelas geradas
│   └── tcc_texto/         # Texto do trabalho (docx, LaTeX ou PDF)
│
├── requirements.txt       # Lista de dependências do Python
├── README.md              # Descrição geral do projeto
└── .gitignore             # Arquivos e pastas para ignorar no Git