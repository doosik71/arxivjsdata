# Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting

Aobo Liang, Xingguo Jiang, Yan Sun, Xiaohou Shi, Ke Li (2024)

## 🧩 Problem to Solve

본 논문은 장기 시계열 예측(Long-term Time Series Forecasting, LTSF)에서 발생하는 고유한 문제들을 해결하고자 한다. LTSF는 미래의 트렌드와 패턴에 대해 더 긴 통찰력을 제공하지만, 다음과 같은 기술적 난관이 존재한다.

첫째, 장기 의존성(Long-term dependencies) 포착의 어려움이다. 시계열 데이터는 데이터의 비정상성(non-stationarity), 노이즈, 이상치 등으로 인해 장기적인 패턴을 학습하기 어렵다. 또한, 시점별(point-wise) 토큰은 시맨틱 정보 밀도가 낮아 중복 노이즈를 유발할 가능성이 크다.

둘째, 계산 효율성과 예측 성능 사이의 트레이드-오프 문제이다. 기존의 Transformer 기반 모델들은 Self-attention 메커니즘의 이차 복잡도(quadratic complexity)로 인해 계산 자원 소모가 극심하며, 이는 학습 및 추론 속도 저하로 이어진다.

셋째, 변수 간의 상호작용(inter-series dependencies)과 개별 변수 내의 진화 패턴(intra-series dependencies) 중 어느 것에 집중할 것인지에 대한 결정 문제이다. 데이터셋의 특성에 따라 채널 독립적(channel-independent) 전략이 유리할 때가 있고, 채널 혼합(channel-mixing) 전략이 유리할 때가 있으나, 이를 자동으로 결정하는 명확한 기준이 부족했다.

따라서 본 논문의 목표는 Mamba라는 State Space Model(SSM)을 기반으로 계산 효율성을 유지하면서도, 시계열 데이터의 복잡한 의존성을 효과적으로 캡처할 수 있는 Bi-Mamba+ 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 Mamba의 선택적 스캔 능력을 유지하면서 시계열 데이터의 특성을 반영할 수 있도록 구조를 개선하는 것이다.

1. **Mamba+ 블록 설계**: 기존 Mamba 블록에 Forget gate를 추가하여, 새로운 특징과 과거의 특징을 보완적인 방식으로 선택적으로 결합함으로써 더 긴 범위의 역사적 정보를 보존하도록 하였다.
2. **Bi-Mamba+ 구조**: Mamba+ 블록을 순방향(forward)과 역방향(backward)으로 모두 적용하여, 시계열 요소 간의 상호작용을 더 포괄적으로 모델링하고 모델의 강건성을 높였다.
3. **SRA (Series-Relation-Aware) Decider**: Spearman 상관 계수를 활용하여 데이터셋의 변수 간 상관관계를 측정하고, 이에 따라 채널 독립적(CI) 토큰화 전략과 채널 혼합(CM) 토큰화 전략 중 최적의 방법을 자동으로 선택하는 메커니즘을 도입하였다.
4. **Patch-wise Tokenization**: 시계열 데이터를 패치 단위로 나누어 토큰화함으로써 계산 복잡도를 낮추고, 더 풍부한 시맨틱 정보를 추출하여 세밀한 입도(fine granularity)에서 장기 의존성을 학습하게 하였다.

## 📎 Related Works

### 시계열 예측 연구
기존의 Transformer 기반 모델들(Informer, Autoformer, Pyraformer, FEDformer 등)은 Self-attention을 통해 장기 의존성을 포착하려 했으나, 연산 복잡도 문제가 지속되었다. PatchTST는 패칭(patching)과 채널 독립적 전략을 통해 성능을 높였고, Crossformer는 차원 간 어텐션을 통해 변수 간 의존성을 모델링하였다. iTransformer는 어텐션 층을 반전시켜 변수 간 의존성을 직접 모델링하는 성과를 거두었으나, 전체 시퀀스를 단순 MLP로 매핑하여 세부적인 진화 패턴을 놓치는 한계가 있다.

### SSM 기반 모델
SSM은 CNN의 병렬 학습 능력과 RNN의 빠른 추론 속도를 결합한 구조이다. 최근 제안된 Mamba는 선택적 스캔 메커니즘을 통해 긴 시퀀스 모델링에서 탁월한 성능을 보였다. 이를 시계열에 적용한 S-Mamba는 변수 간 의존성 포착에 집중했으나 토큰화 방식이 단순했고, MambaMixer는 양방향 구조를 채택했으나 게이팅 브랜치가 새로운 특징 추출을 방해할 수 있는 가능성이 있었다. TimeMachine은 통합 구조를 제안했으나 전략 선택 기준이 단순히 데이터셋의 길이와 변수 수에 의존하여 데이터의 통계적 특성을 충분히 고려하지 못했다.

## 🛠️ Methodology

### 전체 시스템 구조
Bi-Mamba+의 전체 파이프라인은 다음과 같은 순서로 진행된다.
$\text{입력 시계열} \rightarrow \text{Instance Normalization (RevIN)} \rightarrow \text{SRA Decider (전략 결정)} \rightarrow \text{Patching \& Tokenization} \rightarrow \text{Bi-Mamba+ Encoders} \rightarrow \text{Flatten Head \& Linear Projector} \rightarrow \text{Instance Denorm} \rightarrow \text{최종 예측값}$

### 주요 구성 요소 및 상세 설명

#### 1. Instance Normalization
시계열의 비정상성(non-stationarity) 문제를 해결하기 위해 RevIN을 사용한다. 이는 입력 데이터를 정규화하여 분포 변화를 제거하고, 모델의 출력 단계에서 다시 역정규화(denormalization)를 수행하여 원래의 스케일로 복원한다.

#### 2. SRA (Series-Relation-Aware) Decider
데이터셋의 특성에 맞게 토큰화 전략을 자동으로 결정한다. Spearman 상관 계수 $\rho_{i,j}$를 사용하여 변수 쌍 간의 단조 관계를 측정한다.
$$\rho_{i,j} = 1 - \frac{6 \sum_{k=0}^{n} (\text{Rank}(t_{i,k}) - \text{Rank}(t_{j,k}))^2}{n(n^2-1)}$$
여기서 $n$은 관측 샘플 수이다. 이후 임계값 $\lambda$와 $0$을 기준으로 양의 상관관계를 가진 변수 쌍의 최대 개수 $\rho_{\lambda \max}$와 $\rho_{0 \max}$를 구하고, 관계 비율 $r = \rho_{\lambda \max} / \rho_{0 \max}$를 계산한다. $r \ge 1-\lambda$이면 채널 혼합(channel-mixing) 전략을, 그렇지 않으면 채널 독립적(channel-independent) 전략을 선택한다.

#### 3. Tokenization Process
입력 시퀀스를 길이 $P$인 패치들로 나눈다.
- **Channel-Independent (CI)**: 각 단변량 시퀀스를 독립적으로 패칭하여 토큰 $E_{ind} \in \mathbb{R}^{M \times J \times D}$를 생성한다.
- **Channel-Mixing (CM)**: 동일한 인덱스의 패치들을 묶어 토큰화 층을 통과시켜 $E_{mix} \in \mathbb{R}^{J \times M \times D}$를 생성한다.

#### 4. Mamba+ Block
기존 Mamba의 게이팅 구조가 근접 정보에 치우치는 경향을 해결하기 위해 Forget gate를 도입하였다.
- $x'$: 1D Convolution의 결과물
- $z$: 게이트 브랜치의 결과물
- $y$: SSM 블록의 결과물

최종 출력 $y'$는 다음과 같이 계산된다.
$$y' = y \otimes \text{SiLU}(z) + x' \otimes (1 - \sigma(z))$$
여기서 $(1 - \sigma(z))$가 Forget gate 역할을 하며, SSM을 통해 얻은 새로운 특징($y$)과 1D Conv를 통해 얻은 원시 특징($x'$)을 보완적으로 결합하여 역사적 정보를 더 오래 보존한다.

#### 5. Bi-Mamba+ Encoder
단방향 모델의 한계를 극복하기 위해 순방향과 역방향 두 개의 Mamba+ 블록을 배치한다. 
- 입력을 그대로 사용하는 순방향 경로와, 시퀀스를 뒤집어(Flip) 처리하는 역방향 경로의 결과를 합산한다.
- 이후 Transformer의 인코더와 유사하게 잔차 연결(Residual Connection), Layer Normalization, Feed-Forward Network를 거쳐 최종 특징을 도출한다.

#### 6. 학습 목표 및 손실 함수
모델은 예측값 $\hat{Y}$와 실제값 $Y$ 사이의 평균 제곱 오차(Mean Squared Error, MSE)를 최소화하도록 학습된다.
$$L(Y, \hat{Y}) = \frac{1}{|Y|} \sum_{i=1}^{|Y|} (y^{(i)} - \hat{y}^{(i)})^2$$

## 📊 Results

### 실험 설정
- **데이터셋**: Weather, Traffic, Electricity, Solar 및 4개의 ETT 데이터셋 (총 8개).
- **비교 대상**: Autoformer, PatchTST, Crossformer, iTransformer (Transformer 계열), DLinear (MLP 계열), TimesNet (CNN 계열), WITRAN (RNN 계열), CrossGNN (GNN 계열), S-Mamba (SSM 계열).
- **측정 지표**: MSE (Mean Squared Error), MAE (Mean Absolute Error).
- **예측 길이 ($H$)**: 96, 192, 336, 720.

### 주요 결과
1. **정량적 성능**: 모든 데이터셋과 예측 길이에서 Bi-Mamba+가 가장 우수한 성능을 보였다. 특히 SOTA 모델인 iTransformer 대비 MSE는 평균 4.72%, MAE는 2.60% 감소하였다. SSM 기반의 S-Mamba와 비교해서도 MSE 3.76%, MAE 2.67%의 성능 향상을 이루었다.
2. **효율성 분석**: 변수 수가 많은 Traffic 데이터셋에서 Self-attention 기반의 iTransformer는 메모리 사용량이 이차적으로 급증하는 반면, Bi-Mamba+는 SSM의 선형 복잡도 덕분에 메모리 사용량이 선형적으로 증가하며 학습 속도 또한 더 빨랐다.
3. **SRA Decider의 유효성**: 분석 결과, 변수가 적은 ETT 데이터셋에서는 CI 전략을, 변수가 많은 Weather, Traffic, Electricity 데이터셋에서는 CM 전략을 선택하여 최적의 성능을 냈음을 확인하였다.
4. **장기 의존성 포착**: Look-back window ($L$) 크기를 늘렸을 때, Bi-Mamba+는 다른 모델들보다 더 일관되게 예측 성능이 향상되는 모습을 보여 장기 의존성 포착 능력이 뛰어남을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 해석
Bi-Mamba+의 성능 향상은 단순히 모델 크기를 키운 것이 아니라, 시계열 데이터의 특성에 맞는 **적응형 토큰화(SRA Decider)**와 **정보 보존 메커니즘(Mamba+ Block)**, 그리고 **양방향 맥락 파악(Bi-Mamba+ Encoder)**을 유기적으로 결합한 결과이다. 특히, 기존 Mamba가 가진 '최근 정보 편향' 문제를 Forget gate 도입으로 해결하여 LTSF에 적합한 형태로 개선한 점이 인상적이다.

### 한계 및 논의사항
본 논문에서는 8개의 공개 데이터셋에 대해 검증하였으나, 실제 산업 현장의 더 복잡하고 동적인 시나리오(예: 네트워크 트래픽의 급격한 변동)에서의 일반화 성능에 대해서는 추가적인 검증이 필요하다. 또한, SRA Decider에서 사용한 임계값 $\lambda=0.6$이 보편적으로 최적인지에 대한 이론적 근거보다는 실험적 결과에 의존하고 있다는 점이 한계로 지적될 수 있다.

### 비판적 해석
Mamba 기반 모델들이 Transformer의 강력한 대안으로 떠오르고 있지만, 시계열 데이터에서는 단순한 구조의 DLinear와 같은 MLP 모델들이 의외로 강력한 성능을 내는 경우가 많다. Bi-Mamba+가 이들보다 성능이 좋다는 점은 고무적이나, 복잡도가 증가한 만큼의 성능 이득이 모든 도메인에서 절대적인지, 혹은 특정 복잡도를 가진 데이터셋에서만 유리한지에 대한 심층 분석이 보완된다면 더 설득력이 있을 것이다.

## 📌 TL;DR

본 논문은 장기 시계열 예측(LTSF)을 위해 **양방향 Mamba 구조에 Forget gate와 적응형 토큰화 전략을 결합한 Bi-Mamba+**를 제안한다. Spearman 상관 계수 기반의 SRA Decider를 통해 데이터 특성에 맞는 최적의 토큰화 방식(CI vs CM)을 자동으로 선택하며, 패칭(Patching)과 양방향 SSM 인코더를 통해 계산 효율성과 예측 정확도를 동시에 확보하였다. 실험 결과, 기존 Transformer 및 SSM 기반 SOTA 모델들을 성능과 메모리 효율성 측면 모두에서 능가하였으며, 이는 향후 고차원 대규모 시계열 데이터 분석 및 예측 연구에 중요한 기여를 할 것으로 보인다.