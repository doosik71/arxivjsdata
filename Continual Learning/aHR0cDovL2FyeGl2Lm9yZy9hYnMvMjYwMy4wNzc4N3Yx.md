# VISION TRANSFORMERS THAT NEVER STOP LEARNING

Caihao Sun, Minqi Yuan, Shiyuan Wang, Jiayu Chen (2026)

## 🧩 Problem to Solve

본 논문은 지속 학습(Continual Learning) 환경에서 모델이 새로운 태스크에 적응하는 능력을 점진적으로 상실하는 **Plasticity Loss(가소성 상실)** 문제를 다룬다. 가소성 상실은 모델이 학습을 지속함에 따라 새로운 개념을 학습하는 능력이 저하되는 현상으로, 인공 일반 지능(AGI) 구현을 위한 핵심 과제인 평생 학습(Lifelong Learning)의 근본적인 장애물이다.

기존의 가소성 상실 연구는 주로 다층 퍼셉트론(MLP)이나 합성곱 신경망(CNN)과 같은 균질한(homogeneous) 구조에서 이루어졌다. 그러나 현대 딥러닝의 중추가 된 Vision Transformer(ViT)와 같이 구조적으로 이질적인(heterogeneous) 어텐션 기반 모델에서 가소성 상실이 어떻게 발생하는지에 대한 메커니즘은 충분히 연구되지 않았다. 따라서 본 논문의 목표는 ViT에서의 가소성 상실 현상을 체계적으로 진단하고, 이를 완화하여 모델의 학습 능력을 지속적으로 유지하는 방법론을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **ViT 가소성 상실의 체계적 진단**: 장기적인 비정상 상태(non-stationary) 태스크 스트림에서의 실험을 통해, ViT가 Feed-Forward Network(FFN) 모듈 내의 유효 랭크(effective rank) 급락과 휴면 유닛(dormant units)의 증가라는 독특한 형태의 가소성 상실을 겪음을 밝혀냈다.
2. **기존 완화 전략의 한계 분석**: 구조적 재초기화(structural re-initialization) 및 최적화 기반 방법들을 평가하여, 단순한 뉴런 교체 방식으로는 ViT의 복잡한 멀티모달 랜드스케이프에서 요구되는 표현 다양성을 회복하기 어렵다는 점을 확인하였다.
3. **ARROW 옵티마이저 제안**: 그래디언트 방향의 집중 현상을 해결하기 위해 온라인 윈도우 공분산(online windowed covariance)을 이용한 저랭크 곡률 프록시(low-rank curvature proxy) 기반의 geometry-aware 옵티마이저인 **ARROW(Adaptive Rank-Reshaping via Online Windowed covariance)**를 제안하였다.

## 📎 Related Works

### 지속 학습 및 가소성 상실

지속 학습의 핵심은 과거의 지식을 유지하는 '안정성(Stability)'과 새로운 정보를 수용하는 '가소성(Plasticity)' 사이의 딜레마를 해결하는 것이다. 기존 연구들은 주로 정규화, 리플레이, 최적화 방법을 통해 파괴적 망각(catastrophic forgetting)을 방지하는 안정성 확보에 집중해 왔다. 최근에는 가소성 상실을 완화하기 위해 뉴런을 재초기화하는 CBP(Continual Back-propagation)나 학습률을 동적으로 조절하는 TRAC와 같은 옵티마이저 연구가 진행되었다.

### ViT의 특성 및 한계

ViT는 MHSA(Multi-Head Self-Attention)와 FFN으로 구성된 이질적 구조를 가진다. 기존 연구에 따르면 어텐션 모듈은 초기 헤드 전문화(early head specialization) 경향이 있으며, FFN은 특징 포화(feature saturation)와 표현 붕괴(representation collapse)를 겪는 것으로 알려져 있다. 하지만 이러한 구성 요소들이 계층적으로 어떻게 가소성 상실에 기여하는지에 대한 상세한 진단은 부족한 상태였다.

## 🛠️ Methodology

### 1. 가소성 상실 진단 지표

논문은 가소성을 정량화하기 위해 다음과 같은 지표를 사용한다.

- **AAT (Average Accuracy across Tasks)**: 전체 태스크에 대한 평균 정확도.
- **Effective Rank ($\text{erank}$)**: 특성 공간의 차원성을 측정하여 표현 공간의 붕괴 여부를 확인한다.
- **FAU (Fraction of Active Units)**: FFN 내에서 활성화된 뉴런의 비율을 측정하여 '죽은 뉴런'의 비중을 파악한다.

### 2. ARROW 옵티마이저의 설계 원리

기존의 TRAC와 같은 방법은 학습률(step-size)만을 조절하지만, 가소성 상실은 본질적으로 그래디언트 방향이 과거 태스크에 의해 형성된 지배적인 방향으로 쏠리는 **기하학적 문제(geometric issue)**이다. ARROW는 2차 최적화(Newton update)의 개념을 차용하여 곡률이 큰 방향(지배적인 방향)의 업데이트는 억제하고, 곡률이 작은 방향(소외된 방향)의 업데이트는 증폭시켜 표현 공간의 랭크 붕괴를 막는다.

### 3. 상세 알고리즘 및 방정식

ARROW는 다음과 같이 파라미터 업데이트를 수행한다.

$$\Delta\theta_t = -\eta_t (\alpha_t I + \beta C_t)^{-1} g_t$$

여기서 각 변수의 의미는 다음과 같다.

- $g_t$: 현재 시점의 그래디언트.
- $C_t$: 윈도우 기반의 그래디언트 공분산 추정치로, 최근 $W$개 시점의 그래디언트를 이용해 계산한다.
  $$C_t = \frac{1}{W} \sum_{i=t-W+1}^{t} g_i g_i^\top$$
- $\alpha_t$: 댐핑 팩터(damping factor)로 수치적 안정성을 제공한다.
- $\beta$: 곡률 보정 강도를 조절하는 하이퍼파라미터이다.
- $\eta_t$: 학습률이다.

**계산 효율성 최적화**:
$C_t$는 매우 큰 행렬이므로 직접 역행렬을 계산하는 것은 불가능하다. ARROW는 $C_t$가 최대 랭크 $W$를 가지는 저랭크 구조임을 이용하여 **Woodbury identity**를 적용함으로써 계산 복잡도를 획기적으로 낮춘다.

$$\text{Update} = \frac{1}{\alpha_t} g_t - \frac{1}{\alpha_t^2} U_t \left( I + \frac{1}{\alpha_t} U_t^\top U_t \right)^{-1} U_t^\top g_t$$

여기서 $U_t$는 $\sqrt{\beta/W}$와 그래디언트 행렬의 곱으로 정의된다.

## 📊 Results

### 실험 설정

- **데이터셋**: CIFAR-100 (최대 200개 태스크), ImageNet-R (최대 50개 태스크).
- **백본 모델**: ViT-B/16.
- **비교 대상**: Baseline, CBP, NaP, L2P, TRAC.
- **평가 지표**: AAT (Average Accuracy across Tasks).

### 주요 결과

1. **가소성 상실의 확인**: 모든 모델에서 태스크가 진행됨에 따라 AAT가 하락하는 현상이 관찰되었다. 특히 ViT는 깊은 층으로 갈수록 $\text{erank}$가 급격히 감소하고 FFN의 가소성이 크게 훼손됨을 확인하였다.
2. **최적화 기반 방법의 우위**: CBP와 같은 뉴런 재초기화 방법보다 TRAC와 같은 최적화 조절 방법이 ViT의 가소성 유지에 훨씬 효과적이었다.
3. **ARROW의 성능**: ARROW는 모든 벤치마크에서 가장 높은 AAT를 기록하였다. 특히 태스크 스트림의 후반부로 갈수록 가소성 상실이 심화되는 시점에서 TRAC나 L2P보다 우수한 성능을 보였다. (예: CIFAR-100 25개 태스크에서 ARROW는 73.89%의 AAT를 달성하여 Baseline의 70.93%를 크게 상회함).
4. **적용 범위 분석**: ARROW를 모든 블록에 적용하는 것보다, 가소성 상실이 가장 심하게 일어나는 **마지막 attention 블록들**에 적용했을 때 가장 좋은 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 ViT의 가소성 상실이 단순히 파라미터의 크기 문제가 아니라, **표현 공간의 랭크 붕괴(rank collapse)**라는 기하학적 관점에서 접근하여 이를 해결했다는 점에서 학술적 가치가 높다. 특히 FFN이 구조적 병목(bottleneck)임을 밝혀내고, MHSA의 $V$ 행렬이 태스크 변화에 가장 민감하게 반응한다는 세부 진단 결과는 향후 ViT 기반의 지속 학습 연구에 중요한 기초 자료가 될 것이다.

### 한계 및 비판적 해석

1. **하이퍼파라미터 민감도**: ARROW는 $\alpha, \beta, W$ 등 추가적인 하이퍼파라미터를 도입하며, 실험 결과에서도 $\alpha$ 값에 따라 성능 차이가 크게 나타나는 민감성을 보인다. 이에 대한 자동화된 튜닝 방법이나 강건한 기본값 제안이 부족하다.
2. **계산 비용**: 저랭크 근사를 통해 효율성을 높였으나, 여전히 윈도우 기반의 그래디언트 버퍼를 유지해야 하므로 메모리 오버헤드가 발생한다. (논문에서는 ViT와 유사하다고 주장하나 구체적인 메모리 점유율 수치는 미비하다).
3. **AAT 지표의 객관성**: 저자들도 언급했듯이, 초기 성능(base performance)이 다른 모델 간의 비교에서 AAT가 가소성 향상을 완전히 객관적으로 대변하지 못할 가능성이 있다.

## 📌 TL;DR

본 논문은 Vision Transformer(ViT)가 지속 학습 과정에서 FFN의 랭크 붕괴와 깊은 층의 불안정성으로 인해 새로운 데이터를 학습하는 능력(가소성)을 잃는다는 것을 체계적으로 분석하였다. 이를 해결하기 위해 그래디언트의 공분산을 이용하여 업데이트 방향을 적응적으로 재형성하는 **ARROW** 옵티마이저를 제안하였으며, 이는 기존의 단순 학습률 조절이나 뉴런 재초기화 방식보다 훨씬 효과적으로 ViT의 평생 학습 능력을 유지함을 입증하였다. 이 연구는 향후 거대 모델의 지속적인 적응 및 효율적인 파라미터 업데이트 전략 연구에 중요한 기여를 할 것으로 보인다.
