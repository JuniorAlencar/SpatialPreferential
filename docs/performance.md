# Otimizações do simulador

As alterações em `src/SamuraI.cpp` preservam a distribuição espacial, o gerador
aleatório, a ligação preferencial sem reposição, as definições das propriedades
e os blocos do jackknife. Os caminhos mínimos continuam sendo calculados para
todos os pares, sem amostragem.

## Geração e memória

Para cada novo vértice, os pesos `grau * distância²^(-alpha_a/2)` são calculados
uma vez. Após escolher uma conexão, somente o grau do vizinho escolhido muda;
esse vizinho fica excluído dos próximos sorteios. Portanto, os pesos dos demais
candidatos continuam válidos. Os somatórios e as divisões mantêm a ordem do
código anterior, assim como as chamadas ao gerador aleatório. Quando
`alpha_a=0`, o peso é exatamente o grau e dispensa o cálculo geométrico.

Os vetores de pesos e de seleção são reutilizados. Os pesos físicos das arestas
ficam no próprio grafo, substituindo o `unordered_map` que armazenava uma cópia
e não era consultado pelo Dijkstra. A geração continua quadrática em `N`, mas
elimina a repetição dos cálculos geométricos para as `m0` conexões.

## Propriedades

- **Dijkstra:** reutiliza os pesos e constrói uma representação contígua das
  adjacências, com dois arcos por aresta não direcionada. Reutiliza os buffers
  de distâncias e cores entre as fontes.
- **BFS:** utiliza adjacências contíguas, uma fila e um vetor de distâncias
  reutilizados em todas as buscas.
- **Árvores:** quando o grafo tem `N-1` arestas e é conectado, uma aresta que
  separa subárvores de tamanhos `s` e `N-s` participa de `s*(N-s)` caminhos não
  ordenados. Somar essa contribuição, multiplicada pelo peso quando aplicável,
  produz a média exata em `O(N)`. Os diâmetros são obtidos na mesma passagem
  pelas maiores alturas das subárvores. Os pesos devem ser finitos.
- **Clustering:** orienta arestas pela ordem `(grau, id)` e conta cada triângulo
  uma vez, creditando os três vértices. Mantém os coeficientes e a soma em
  `float`, como no código Boost anterior.
- **Spearman:** calcula os mesmos ranks médios para empates usando histogramas
  de graus inteiros, evitando ordenações em cada bloco. A correlação continua
  usando a mesma ordem das arestas.
- **Jackknife:** reutiliza buffers e percorre uma lista contígua de extremos
  das arestas. Os graus são recontados no subgrafo de cada bloco, como antes.

O caminho especial para árvores muda a ordem das somas dos pesos. Isso pode
causar diferenças nos últimos bits de ponto flutuante, sem mudar a grandeza
calculada. Os testes usam tolerância absoluta/relativa de `2e-12` somente para
a média e o diâmetro ponderados das árvores na comparação com a versão antiga;
as demais propriedades devem coincidir exatamente.

Para `m0>=2`, o cálculo exato de caminhos entre todos os pares ainda domina o
custo em redes grandes: BFS custa `O(N*(N+M))` e Dijkstra tem limite
`O(N*(N+M)*log N)`. Os ganhos medidos em redes pequenas não eliminam esse custo
para `N=100000`. O programa continua usando uma única thread por rede, permitindo
que os scripts existentes executem redes independentes em paralelo.

## Resultados medidos

Ubuntu 24.04, GCC 13.3, Boost 1.83, compilação `-O3 -DNDEBUG`, uma thread.
Para `N=2500`, dimensão 2, `alpha_g=2`, seed 1234 e 100 blocos de jackknife,
os tempos abaixo são medianas de três execuções. O total é a soma das etapas
de geração, propriedades e gravação do GML, sem incluir a inicialização do processo.

| Configuração | Total anterior | Total otimizado | Aceleração total | Aceleração da geração |
| --- | ---: | ---: | ---: | ---: |
| `m0=1`, `alpha_a=2` | 0,679 s | 0,078 s | 8,70× | 1,15× |
| `m0=2`, `alpha_a=2` | 1,128 s | 0,812 s | 1,39× | 2,30× |
| `m0=5`, `alpha_a=0` | 1,648 s | 1,078 s | 1,53× | 5,59× |

Nas duas configurações com ciclos, o Dijkstra ficou aproximadamente 1,19× mais
rápido; BFS, 1,76–1,96×; Spearman, 4,47–4,95×. O clustering ficou 2,19× e 7,70×
mais rápido, respectivamente. Em árvores pequenas, o novo clustering custou
ligeiramente mais (0,148 ms contra 0,110 ms); o ganho total vem principalmente
dos caminhos calculados em tempo linear.

Também foi medido o executável completo com `run_mode=2`, `N=12000`, `m0=5`,
`alpha_a=0`, dimensão 2, `alpha_g=2` e seed 1234. Em uma execução por versão,
`/usr/bin/time` registrou 5,50 s → 1,08 s e pico de RSS de 14384 KiB → 11696 KiB
(aproximadamente 19% menos memória). O GML descomprimido permaneceu idêntico.
Essas medições descrevem esses cenários e não são uma garantia de aceleração
igual para outras redes ou máquinas.

## Validação e reprodução

```bash
cmake -S . -B build
cmake --build build -j2
python3 tests/verify_optimizations.py
```

Também foi executada a mesma bateria com AddressSanitizer, UndefinedBehaviorSanitizer
e LeakSanitizer, sem erros. Para reproduzir em um terminal local:

```bash
python3 tests/verify_optimizations.py --sanitize
```

O teste compila um executável temporário e confere 44 configurações, incluindo
dimensões 1–4, `m0=1,2,5`, cliques completos, diferentes expoentes, pesos zero,
fallback aleatório por underflow, graus excedentes e diferentes números de
blocos. Uma implementação Python independente confere todas as propriedades.
Os modos de execução 1, 2 e 3 também foram comparados pelo executável principal
com `N=64`, `m0=2`, dimensão 2, `alpha_a=alpha_g=2` e seed 1234: os CSVs e os
GMLs descomprimidos coincidiram com a referência.

Para comparar com uma cópia de `src` anterior às alterações, incluindo o GML
descomprimido byte a byte e a mediana de três execuções por cenário:

```bash
python3 tests/verify_optimizations.py --baseline-src /caminho/src-anterior --benchmark
```

A referência utilizada foi `src/` do commit `8384998`. Uma cópia pode ser
extraída sem modificar os arquivos de trabalho:

```bash
baseline_dir=$(mktemp -d)
git archive 8384998 src | tar -x -C "$baseline_dir"
python3 tests/verify_optimizations.py --baseline-src "$baseline_dir/src" --benchmark
```

Os arquivos de entrada, redes e executáveis de teste ficam em um diretório
temporário. O código de produção é compilado normalmente em `build/exe1`.
