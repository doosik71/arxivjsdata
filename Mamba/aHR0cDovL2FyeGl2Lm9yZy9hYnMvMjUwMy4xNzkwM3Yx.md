# GLADMamba: Unsupervised Graph-Level Anomaly Detection Powered by Selective State Space Model

Yali Fu, Jindong Li, Qi Wang, and Qianli Xing (2025)

## 🧩 Problem to Solve

본 논문은 비지도 그래프 레벨 이상치 탐지(Unsupervised Graph-Level Anomaly Detection, UGLAD) 문제를 해결하고자 한다. UGLAD는 사회 관계망 분석, 항암제 발견, 독성 분자 식별 등 다양한 도메인에서 중요한 과제이며, 레이블이 없는 상태에서 대다수의 정상 그래프와 확연히 다른 패턴을 보이는 소수의 이상치 그래프를 식별하는 것을 목표로 한다.

기존의 UGLAD 방법론들은 다음과 같은 세 가지 핵심적인 한계를 가지고 있다.

1. **장거리 의존성 모델링의 어려움**: 대부분의 GNN 기반 방법론은 Over-squashing 문제로 인해 그래프 내 멀리 떨어진 노드 간의 의존성을 효율적으로 캡처하지 못한다.
2. **계산 복잡도 문제**: Transformer 기반의 어텐션 메커니즘을 도입하여 위 문제를 해결하려 했으나, 계산 복잡도가 입력 크기에 대해 제곱(quadratic)으로 증가하여 대규모 그래프에 적용하기 어렵다.
3. **스펙트럼 정보의 부재**: 그래프의 스펙트럼 에너지 분포가 정상과 이상치 간에 차이가 있음에도 불구하고, 기존 연구들은 주로 공간 도메인(spatial domain)의 정보에만 의존하며 스펙트럼 정보를 명시적으로 활용하지 않는다.

따라서 본 논문의 목표는 선형 복잡도로 장거리 의존성을 캡처할 수 있는 선택적 상태 공간 모델(Selective State Space Model, Mamba)을 UGLAD에 도입하고, 스펙트럼 정보를 통합하여 탐지 성능과 효율성을 동시에 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mamba의 선택적 상태 전이 메커니즘을 활용하여 이상치 탐지에 중요한 정보만을 동적으로 선택하고, 그래프의 스펙트럼 특성을 모델의 파라미터 생성 과정에 직접 반영하는 것이다.

주요 기여 사항은 다음과 같다.

- **GLADMamba 프레임워크 제안**: Mamba를 UGLAD 분야에 최초로 도입하여 선형 시간 복잡도로 효율적인 그래프 모델링을 구현하였다.
- **View-Fused Mamba (VFM) 설계**: 서로 다른 뷰(특징 뷰와 구조 뷰)의 정보를 상호 보완적으로 융합하기 위해 Mamba-Transformer 스타일의 아키텍처를 설계하였다.
- **Spectrum-Guided Mamba (SGM) 설계**: Rayleigh quotient를 통해 얻은 스펙트럼 에너지를 Mamba의 시스템 파라미터 결정 과정에 투입함으로써, 공간 도메인과 스펙트럼 도메인의 상호작용을 유도하고 이상치 관련 패턴을 정교하게 정제한다.

## 📎 Related Works

### 1. 그래프 레벨 이상치 탐지

기존 방법론은 크게 두 단계 방식(Two-stage)과 엔드투엔드 방식(End-to-end)으로 나뉜다.

- **Two-stage**: Weisfeiler-Lehman (WL) 커널이나 Propagation Kernel (PK)을 사용하여 그래프 임베딩을 먼저 추출한 후, Isolation Forest (iF)나 One-class SVM 같은 전통적인 이상치 탐지 알고리즘을 적용한다.
- **End-to-end**: GNN을 백본으로 사용하여 표현 학습과 탐지를 동시에 수행하는 방법으로, GOOD-D(계층적 대조 학습)나 CVTGAD(단순화된 Transformer 기반) 등이 있다. 하지만 앞서 언급한 것처럼 Over-squashing이나 계산 복잡도 문제가 여전히 존재한다.

### 2. 상태 공간 모델 (State Space Models)

SSM은 연속적인 시스템의 동적 진화를 모델링하는 프레임워크이며, 최근 Mamba는 파라미터를 입력에 의존하게 만드는 선택 메커니즘(selection mechanism)을 도입하여 시퀀스 모델링에서 뛰어난 성능과 효율성을 입증하였다. 본 논문은 이러한 Mamba의 강점을 그래프 데이터의 특성에 맞게 변형하여 UGLAD에 적용함으로써 기존 GNN 및 Transformer 기반 방식의 한계를 극복하고자 한다.

## 🛠️ Methodology

### 1. 데이터 증강 및 인코딩

모델은 입력 그래프에 대해 특징(feature) 중심의 원본 뷰($view_o$)와 구조(structure) 중심의 증강 뷰($view_a$)를 생성한다. 두 뷰는 각각 독립적인 GNN 인코더를 통과하여 노드 임베딩 $H^o, H^a$와 그래프 임베딩 $h^o_G, h^a_G$를 생성한다.

### 2. View-Fused Mamba (VFM)

VFM은 서로 다른 두 뷰의 정보를 융합하는 모듈이다. 특이한 점은 한 뷰의 입력을 사용하여 다른 뷰의 Mamba 파라미터를 결정하는 상호 의존적 구조를 가진다는 것이다.

- **선택적 파라미터화**: $H^o$와 $H^a$를 각각 LayerNorm, Linear, Conv1d, SiLU 층에 통과시켜 입력값($H^{o}_{input}, H^{a}_{input}$)을 만든다. 이후, 뷰 $o$의 파라미터 $(B^o, C^o, \Delta^o)$는 뷰 $a$의 입력값 $H^{a}_{input}$으로부터 생성되며, 반대 경우도 마찬가지이다.
  - 예: $B^o = W^B_o H^a_{input}, \Delta^o = \text{softplus}(W^{\Delta}_o H^a_{input})$
- **이산화 및 상태 업데이트**: Zero-Order Hold (ZOH) 규칙을 사용하여 연속 파라미터를 이산 파라미터 $(\bar{A}, \bar{B})$로 변환한 후, SSM 상태 전이 식을 통해 출력 $y_{ssm}$을 생성한다.
- **최종 출력**: $y_{ssm}$에 gating 메커니즘($u = \text{SiLU}(\text{Linear}(\text{LayerNorm}(H)))$)을 적용하고 잔차 연결(residual connection)과 LayerNorm을 거쳐 최종 융합 표현 $Z^o, Z^a$를 얻는다.

### 3. Spectrum-Guided Mamba (SGM)

SGM은 그래프의 스펙트럼 정보를 활용하여 임베딩을 정제한다.

- **Rayleigh Quotient**: 그래프의 스펙트럼 에너지를 측정하기 위해 Rayleigh Quotient $R(L, X)$를 사용한다. 이는 고주파 성분이 많을수록 값이 커지며, 이상치 그래프일수록 고주파 영역으로 에너지가 이동한다는 특성을 이용한다.
  $$R(L, X) = \frac{X^T L X}{X^T X} = \frac{\sum_{(i,j) \in E} (x_i - x_j)^2}{\sum_{i \in V} x_i^2}$$
- **스펙트럼 가이드 파라미터**: Rayleigh quotient의 대각 성분을 MLP에 통과시켜 $h_{RQ}$를 얻고, 이를 사용하여 Mamba의 파라미터 $(B, C, \Delta)$를 생성한다. 이를 통해 모델은 스펙트럼 특성에 따라 상태 업데이트를 동적으로 조절하며 이상치 관련 정보에 집중한다.

### 4. 학습 및 추론 절차

- **학습 (Training)**: 두 뷰 사이의 일관성을 최대화하기 위해 노드 레벨과 그래프 레벨에서 InfoNCE 대조 손실(Contrastive Loss)을 사용한다.
  - 노드 레벨 손실 $\mathcal{L}'_{node}$와 그래프 레벨 손실 $\mathcal{L}'_{graph}$를 합산하며, 각 데이터셋의 민감도를 고려하여 표준편차 $\sigma$를 이용한 적응형 손실 함수 $\mathcal{L} = (\sigma_{node})^\alpha \mathcal{L}'_{node} + (\sigma_{graph})^\alpha \mathcal{L}'_{graph}$를 최소화한다.
- **추론 (Inference)**: 모델은 정상 그래프의 공통 패턴을 학습하므로, 이상치 그래프가 입력되면 손실 값 $\mathcal{L}$이 크게 나타난다. 따라서 z-score 표준화를 적용한 $\mathcal{L}_{node}$와 $\mathcal{L}_{graph}$의 합을 최종 이상치 점수 $S$로 사용한다.
  $$S = \text{Std}(\mathcal{L}_{node}) + \text{Std}(\mathcal{L}_{graph})$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: TuDataset 벤치마크의 12개 실제 데이터셋(분자, 생물정보학, 사회 네트워크 등)을 사용하였다.
- **비교 대상**: PK-iF, WL-iF 등 2단계 방식 5종과 OCGIN, GOOD-D, CVTGAD 등 엔드투엔드 방식 4종을 포함한 총 9개 baseline과 비교하였다.
- **지표**: AUC (Area Under the ROC Curve)를 사용하여 성능을 측정하였다.

### 2. 주요 결과

- **탐지 성능**: GLADMamba는 12개 데이터셋 중 8개에서 최고 성능을 기록하였으며, 평균 순위(Avg. Rank) 1.33으로 모든 baseline을 압도하였다. 특히 엔드투엔드 모델들이 2단계 모델들보다 우수한 성능을 보였는데, 이는 특징 학습과 탐지가 통합 최적화되었기 때문이다.
- **효율성 분석**: Transformer 기반의 CVTGAD와 비교했을 때, 대규모 데이터셋(REDDIT-B, p53)에서 FLOPs와 파라미터 수, GPU 메모리 사용량이 훨씬 적음을 확인하였다. 이는 Mamba의 선형 복잡도가 대규모 그래프 처리에 매우 유리함을 입증한다.
- **절제 연구 (Ablation Study)**: VFM과 SGM 모듈을 각각 제거했을 때 성능이 유의미하게 하락하였다. 특히 SGM을 제거했을 때의 성능 저하가 두드러지는데, 이는 정상/이상치 간의 스펙트럼 차이를 이용하는 것이 매우 효과적임을 시사한다.

## 🧠 Insights & Discussion

본 논문은 Mamba라는 최신 시퀀스 모델링 구조를 그래프 이상치 탐지에 성공적으로 이식하였다. 특히 단순히 구조를 가져온 것이 아니라, **"뷰 간의 상호 의존적 파라미터 생성(VFM)"**과 **"스펙트럼 에너지 기반의 상태 제어(SGM)"**라는 그래프 특화 설계를 도입한 점이 돋보인다.

**강점 및 의의**:

- GNN의 Over-squashing 문제와 Transformer의 계산 복잡도 문제를 동시에 해결하였다.
- 그동안 감독 학습 영역에서 주로 다뤄졌던 스펙트럼 정보를 비지도 학습 기반의 UGLAD에 명시적으로 통합한 첫 번째 시도라는 점에서 학술적 가치가 크다.

**한계 및 논의사항**:

- 본 논문에서는 GCN과 GIN을 기본 인코더로 사용하였으나, 인코더의 성능이 전체 파이프라인에 미치는 영향에 대한 심층 분석은 부족하다.
- Rayleigh quotient를 통한 스펙트럼 가이드가 정확히 어떤 종류의 이상치 패턴을 잡아내는지에 대한 정성적인 분석이 더 보완된다면 모델의 해석력이 높아질 것이다.

## 📌 TL;DR

GLADMamba는 **Selective State Space Model (Mamba)**을 최초로 비지도 그래프 레벨 이상치 탐지(UGLAD)에 적용한 프레임워크이다. **View-Fused Mamba(VFM)**를 통해 다중 뷰 정보를 효율적으로 융합하고, **Spectrum-Guided Mamba(SGM)**를 통해 Rayleigh quotient 기반의 스펙트럼 정보를 상태 업데이트에 반영함으로써 탐지 정확도를 높였다. 실험 결과, 기존 SOTA 모델들보다 우수한 AUC 성능을 보였으며, 특히 대규모 그래프에서 Transformer 대비 월등한 계산 효율성을 입증하여 향후 대규모 그래프 이상치 탐지 연구에 중요한 기반이 될 것으로 보인다.
