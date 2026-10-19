## Preferential Attachment in Euclidian Space
# Networks with preferential attachment immersed in Euclidean space
Implementation of a model proposed by Samurai <i> et. al </i> <p>
  <a href="https://www.nature.com/articles/srep27992">Role of dimensionality in complex networks</a>.
</p>
generating random scale-free networks using a preferential attachment mechanism in Euclidian Space. 

A BFS (Breadth First Search) algorithm was implemented to calculate the shortest paths in the network in C++, along with the calculation of assortativity by degree using an adaptation of the Pearson coefficient by Newman
<p>
  <a href="https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.89.208701">Assortative Mixing in Networks</a>.
</p>


## Requirements
### Ambiente Python e Jupyter (Ubuntu)

As dependências foram levantadas a partir de `scripts/jupyter` e
`scripts/python` (incluindo bibliotecas usadas pelos checkpoints antigos).
Execute na raiz do projeto, em um terminal que permita informar a senha do sudo:

```bash
bash scripts/setup_python_env.sh
source ~/.bashrc
source .venv/bin/activate
jupyter lab scripts/jupyter
```

O script instala as dependências de compilação do Python e do simulador C++,
instala pyenv em `~/.pyenv`, seleciona a versão estável mais recente da série
Python 3.12 conhecida pelo pyenv e a registra em `.python-version`.
Cria `.venv`, instala `requirements.txt`, registra o kernel
**Python (SpatialPreferential)** e salva as versões em `requirements.lock.txt`.
Selecione esse kernel no Jupyter ou no VS Code. Execute os notebooks a partir
de `scripts/jupyter` para resolver os imports locais `src` e caminhos relativos.

Para reproduzir as versões de pacotes após a primeira instalação:

```bash
.venv/bin/python -m pip install -r requirements.lock.txt
```

Os notebooks `assortativity.ipynb` e `properties_old.ipynb` importam
`src.FunctionsProperties`, que não está presente no repositório. Esse módulo
precisa ser recuperado para executar as células que dependem dele.

- cmake

- g++ compile

- nlohmann library (json files)

- boost library

- eigen3lib library

- zlib library


# To install *cmake* (linux ubuntu)
```bash
sudo apt install cmake  
```

# To install *nlohmann* (linux ubuntu)
```bash
sudo apt install nlohmann-json3-dev  
```

# To install *boost* (linux ubuntu)
```bash
sudo apt install libboost-all-dev  
```

# To install *eigen3* (linux ubuntu)
```bash
sudo apt install libeigen3-dev  
```

# To install *zlib* (linux ubuntu)
```bash
sudo apt install zlib1g-dev
```


## Running code

As otimizações, os limites de custo e os testes de equivalência estão descritos
em [docs/performance.md](docs/performance.md).

### Geração paralela de amostras

Em `scripts/python/multithread_pc.py`, `compute_dijkstra` e
`compute_spearman` controlam os cálculos opcionais (ambos começam em `False`).
Por padrão, as propriedades calculadas são BFS, Newman e clustering. Para
ativar um opcional, defina a variável correspondente como `True` e gere os
JSONs novamente. O binário também aceita essas duas chaves diretamente no
JSON de entrada; colunas opcionais não calculadas ficam vazias no CSV.

```bash
python3 scripts/python/multithread_pc.py
bash scripts/N_100000_multithread_pc.sh
```

O primeiro job real mede o pico de RAM com `/usr/bin/time` e conta como uma das
amostras solicitadas. O executor calcula o limite de jobs simultâneos com a RAM
total da máquina, os CPUs disponíveis, uma reserva de 15% para o sistema e 30%
de margem sobre o maior pico observado. Ele também verifica a RAM disponível
antes de cada lançamento, considera limites de cgroup, atualiza a estimativa
após cada job e inicia o próximo assim que houver vaga. O relatório fica em
`scripts/benchmark_N_100000.json`.

O terminal mostra a etapa em execução, o progresso da criação da rede a cada
10% dos sítios e o tempo restante estimado. Na primeira amostra, a estimativa
usa o ritmo observado durante a execução; nas seguintes, usa os tempos medidos
no benchmark para cada etapa, incluindo as propriedades e a gravação dos dados.

O executor conta as amostras já presentes por configuração e modo de execução
e produz somente as que faltam até `NumSamples`. O shell pode receber
`--max-jobs 4` para impor um teto manual, ou `--max-freq 2.5GHz` para o limite
de frequência já previsto anteriormente. Com `run_mode=1` e `N=100000`, o
primeiro benchmark também executa o cálculo exato de propriedades e pode demorar.
Essa etapa usa Python padrão, Linux e GNU time; GNU Parallel não é necessário.

```bash

# Navigate to the folder where the clone was made
cd TsallisNetwork

# Create build folder to generate executable
mkdir build && cd build

# Generate the cmake files
cmake ...

# Generate the executable
make

# The exe1 executable will be generated, the input parameters must be the standard expressed in example.json. To execute the code, we must follow the pattern
path_to_executable/exe1 path_to_json/example.json
```

## Data
The data will be stored in <b>/gml_folde</b> and <b>/prop_folder</b>, where the gml file stores all the link pairs in the network, with the Euclidean distance between the site pairs and the degree of each site. While the prop file stores the value of the properties shortest average network path, network diameter and assortativity by degree
