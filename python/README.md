# Rubik's Cube Neural Solver - Python Training

Este diretório contém o código Python para treinar uma rede neural que resolve o cubo de Rubik usando algoritmo genético.

## Configuração do Ambiente

Este projeto usa **[uv](https://github.com/astral-sh/uv)** para gerenciamento de pacotes Python (muito mais rápido que pip!).

### 1. Instalar uv (se ainda não tiver)

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/Mac:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Criar ambiente virtual (se ainda não existir)

```bash
uv venv
```

### 3. Ativar o ambiente virtual

**Opção 1 - Usando o script auxiliar (Recomendado):**
```powershell
.\activate.ps1
```

**Opção 2 - Ativação manual:**

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.\.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 4. Instalar dependências

```bash
uv pip install -r requirements.txt ruff
```

### 5. Verificar instalação

```bash
python --version
uv pip list
```

## Uso do Ruff

O Ruff é uma ferramenta de linting e formatação extremamente rápida para Python.

### Verificar código (linting)

```bash
ruff check .
```

### Corrigir automaticamente

```bash
ruff check . --fix
```

### Formatar código

```bash
ruff format .
```

### Verificar e formatar tudo

```bash
ruff check . --fix && ruff format .
```

## Comandos Úteis do UV

O **uv** é extremamente rápido e oferece comandos similares ao pip:

### Instalar pacotes

```bash
# Instalar um pacote
uv pip install numpy

# Instalar de requirements.txt
uv pip install -r requirements.txt

# Instalar versão específica
uv pip install torch==2.10.0
```

### Gerenciar pacotes

```bash
# Listar pacotes instalados
uv pip list

# Mostrar informações de um pacote
uv pip show torch

# Desinstalar pacote
uv pip uninstall numpy

# Atualizar pacote
uv pip install --upgrade torch
```

### Gerar requirements.txt

```bash
# Congelar dependências atuais
uv pip freeze > requirements.txt
```

### Por que uv é melhor?

- ⚡ **10-100x mais rápido** que pip
- 🔒 **Resolução de dependências mais confiável**
- 💾 **Cache inteligente** de pacotes
- 🎯 **Compatível com pip** (mesma sintaxe)


## Treinamento do Modelo

### Treinamento básico

```bash
python train.py --population 50 --generations 100 --scramble-depth 5
```

### Parâmetros disponíveis

- `--population`: Tamanho da população (padrão: 50)
- `--generations`: Número de gerações (padrão: 100)
- `--scramble-depth`: Número de movimentos para embaralhar (padrão: 5)
- `--max-steps`: Máximo de movimentos para resolver (padrão: 30)
- `--test-cubes`: Número de cubos de teste por avaliação (padrão: 10)
- `--mutation-rate`: Taxa de mutação (padrão: 0.1)
- `--mutation-strength`: Força da mutação (padrão: 0.3)
- `--crossover-rate`: Taxa de crossover (padrão: 0.7)
- `--elitism`: Número de indivíduos elite (padrão: 5)
- `--hidden1`: Tamanho da primeira camada oculta (padrão: 256)
- `--hidden2`: Tamanho da segunda camada oculta (padrão: 128)
- `--output`: Diretório de saída (padrão: weights)
- `--load`: Carregar checkpoint de arquivo
- `--save-every`: Salvar checkpoint a cada N gerações (padrão: 10)
- `--target-fitness`: Parar quando esta fitness for alcançada
- `--quiet`: Saída mínima

### Exemplo de treinamento avançado

```bash
python train.py \
  --population 100 \
  --generations 500 \
  --scramble-depth 7 \
  --max-steps 50 \
  --test-cubes 20 \
  --mutation-rate 0.15 \
  --hidden1 512 \
  --hidden2 256 \
  --save-every 5
```

### Continuar treinamento de um checkpoint

```bash
python train.py --load weights/run_20260127_205000/checkpoint.json --generations 100
```

## Estrutura do Projeto

```
python/
├── cube/              # Implementação do cubo de Rubik
│   ├── cube_state.py  # Estado e movimentos do cubo
│   └── cube_env.py    # Ambiente de treinamento
├── neural/            # Rede neural
│   ├── network.py     # Arquitetura da rede
│   └── weight_export.py  # Exportação de pesos
├── genetic/           # Algoritmo genético
│   ├── evolution.py   # Evolução da população
│   ├── fitness.py     # Avaliação de fitness
│   └── individual.py  # Indivíduo da população
├── train.py           # Script principal de treinamento
├── requirements.txt   # Dependências
├── pyproject.toml     # Configuração do Ruff
└── .gitignore         # Arquivos ignorados pelo Git
```

## Saída do Treinamento

Os resultados do treinamento são salvos em `weights/run_TIMESTAMP/`:

- `best_weights.json`: Pesos da melhor rede (para uso no frontend)
- `training_stats.json`: Estatísticas de treinamento
- `checkpoint.json`: Checkpoint completo para continuar o treinamento

## Dicas

1. **Começar com scramble-depth baixo**: Comece com 3-5 movimentos e aumente gradualmente
2. **Monitorar o progresso**: Use `--save-every 5` para salvar checkpoints frequentes
3. **Ajustar hiperparâmetros**: Experimente diferentes valores de mutation-rate e population
4. **Usar GPU**: O PyTorch usará GPU automaticamente se disponível (muito mais rápido!)
