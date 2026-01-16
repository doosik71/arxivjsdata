# Source-Free Domain Adaptation for Semantic Segmentation

Yuang Liu, Wei Zhang, Jun Wang

## 🧩 Problem to Solve

의미론적 분할(semantic segmentation)을 위한 컨볼루션 신경망(CNN) 기반 접근 방식은 픽셀 단위 주석 데이터에 크게 의존하지만, 이러한 데이터는 수집에 많은 노동력과 비용이 소요됩니다. 이를 해결하기 위해 UDA(Unsupervised Domain Adaptation)가 제안되었지만, 기존 UDA 방법들은 모델 적응 과정에서 소스 및 타겟 도메인 간의 간극을 줄이기 위해 소스 데이터셋에 대한 완전한 접근을 필수적으로 요구합니다. 그러나 실제 시나리오에서는 소스 데이터셋이 비공개(private)이거나 상업적인 이유로 사용할 수 없는 경우가 많아, 잘 훈련된 소스 모델과 레이블 없는 타겟 도메인 데이터셋만으로 적응을 수행해야 하는 '소스-프리(source-free)' 문제가 발생합니다. 특히, 이미지 수준 작업과 달리 픽셀 수준 작업인 의미론적 분할은 각 픽셀이 여러 클래스를 가질 수 있어, 기존의 소스-프리 UDA 방법들을 직접 적용하기 어렵다는 문제점이 있습니다.

## ✨ Key Contributions

- 소스 데이터나 타겟 레이블에 대한 접근 없이 지식 전달(knowledge transfer)과 모델 적응(model adaptation)을 결합한 의미론적 분할을 위한 최초의 소스-프리 UDA 프레임워크인 **SFDA (Source-Free Domain Adaptation)**를 제안했습니다.
- 분할 작업에 특화된 새로운 **듀얼 어텐션 증류(dual attention distillation, DAD)** 메커니즘을 설계하여 문맥 정보를 효과적으로 전달하고 유지하며, 타겟 도메인의 패치 수준 정보를 활용하기 위한 **인트라-도메인 패치 수준 자가 감독(intra-domain patch-level self-supervision, IPSM)** 모듈을 도입했습니다.
- 합성-실제(synthetic-to-real) 및 교차 도시(cross-city) 분할 시나리오에서 제안하는 프레임워크의 효과를 광범위한 실험을 통해 입증했으며, 소스 데이터를 사용하는 기존 UDA 방법들과 비교하여 경쟁력 있는 성능을 달성했습니다.

## 📎 Related Works

- **의미론적 분할을 위한 UDA**: 기존 UDA 방법들은 주로 적대적 학습(adversarial learning), 이미지-투-이미지 변환(image-to-image translation), 타겟 유사 레이블(pseudo-labels)을 사용한 자가 감독(self-supervision) 등을 활용하여 도메인 간의 불일치를 줄이지만, 모든 방법은 주석이 달린 소스 데이터셋의 가용성을 전제로 합니다.
- **지식 증류(Knowledge Distillation, KD)**: 대규모 교사 네트워크의 지식을 소형 학생 네트워크로 전달하는 기법으로, 모델 압축, 도메인 적응 등 다양한 분야에서 활용됩니다.
- **데이터-프리 지식 증류(Data-Free Knowledge Distillation)**: 데이터 프라이버시 문제로 인해 주목받는 분야로, 원본 훈련 데이터 없이 생성 모델(GAN 등)을 통해 가짜 샘플을 생성하여 교사 모델의 지식을 학생 모델로 전달합니다. 본 연구는 이 데이터-프리 KD 개념을 의미론적 분할의 소스-프리 도메인 적응 문제로 확장했습니다.

## 🛠️ Methodology

SFDA 프레임워크는 크게 두 단계로 구성됩니다: 지식 전달(Knowledge Transfer)과 모델 적응(Model Adaptation).

### 1. 소스-프리 도메인 지식 전달 (Source-Free Domain Knowledge Transfer, SFKT)

- **소스 도메인 추정**: 생성자 $G$는 임의의 노이즈 $z$를 입력받아 소스 도메인과 유사한 가짜 샘플 $\tilde{x}_{s}$를 생성합니다.
- **BNS 제약**: 생성된 $\tilde{x}_{s}$의 특징 분포가 고정된 소스 모델 $\tilde{S}$의 배치 정규화 통계(Batch Normalization Statistics, BNS)와 일치하도록 $L_{BNS}$ 손실을 적용합니다.
  $$ L*{BNS} = \sum*{l} \| \mu*{l}(\tilde{x}*{s}) - \bar{\mu}_{l} \|_{2}^{2} + \sum*{l} \| \sigma*{l}^{2}(\tilde{x}_{s}) - \bar{\sigma}_{l}^{2} \|_{2}^{2} $$
  여기서 $\mu_{l}(\tilde{x}_{s})$와 $\sigma_{l}^{2}(\tilde{x}_{s})$는 $l$번째 레이어의 배치 평균과 분산 추정치이며, $\bar{\mu}_{l}$과 $\bar{\sigma}_{l}^{2}$는 $\tilde{S}$에 저장된 소스 도메인의 통계입니다.
- **의미론적 인식 적대적 지식 전달**: 소스 모델 $\tilde{S}$와 타겟 모델 $S$ 간의 불일치를 활용하여 지식을 전달합니다.
  - **MAE 손실 ($L_{MAE}$)**: 고정된 소스 모델 $\tilde{S}$와 타겟 모델 $S$ (소스 지식 보존을 위해 매개변수를 공유하는)의 출력 간 평균 절대 오차를 계산합니다.
    $$ L*{MAE} = E*{\tilde{x}_{s}} \left( \frac{1}{C} \| S(\tilde{x}_{s}) - \tilde{y}_{s} \|_{1} \right) $$
        여기서 $\tilde{y}_{s} = \tilde{S}(\tilde{x}_{s})$입니다.
  - **듀얼 어텐션 증류 (DAD) 손실**:
    - **$L_{ss}^{DAD}$**: 소스 모델 $\tilde{S}$와 타겟 모델 $S$ 간의 문맥적 관계(contextual relationships) 불일치를 측정합니다. DAM(Dual Attention Module)은 공간 및 채널 어텐션 맵을 추출하여 특징 맵의 장거리 종속성을 포착합니다.
      $$ L*{ss}^{DAD} = E*{\tilde{x}_{s}} \left( \frac{1}{M} \| A(\tilde{F}_{s}(\tilde{x}_{s})) - A(F_{s}(\tilde{x}_{s})) \|_{1} \right) $$
    - **$L_{st}^{DAD}$**: 가짜 소스 데이터와 실제 타겟 데이터의 듀얼 어텐션 맵 분포 간의 KL 발산을 최소화하여, 생성자가 타겟 도메인의 사전 어텐션 정보를 활용하도록 유도합니다.
      $$ L*{st}^{DAD} = E*{\tilde{x}_{s}}[D_{KL}(S(\tilde{F}_{s}(\tilde{x}_{s})), S(F*{t}(x*{t})))] + E*{\tilde{x}*{s}}[D_{KL}(R(\tilde{F}_{s}(\tilde{x}_{s})), R(F_{t}(x_{t})))] $$
- **생성자 ($G$) 최적화**: $G$는 $L_{BNS} - \alpha L_{MAE} - \beta L_{ss}^{DAD} + \tau L_{st}^{DAD}$를 최소화하도록 훈련됩니다.
- **타겟 모델 ($T$) 최적화**: 초기 지식 전달 단계에서는 $\alpha L_{MAE} + \beta L_{ss}^{DAD}$를 최소화하도록 훈련됩니다.

### 2. 자가 감독 모델 적응 (Self-supervised Model Adaptation)

- **인트라-도메인 패치 수준 자가 감독 모듈 (IPSM)**:
  - 타겟 이미지 $x_t$를 $K \times K$ 크기의 패치 $x_{t,k}$로 분할합니다.
  - 각 패치 $x_{t,k}$에 대한 타겟 모델의 예측 맵 $p_{t,k}$의 평균 엔트로피 점수 $E(x_{t,k})$를 계산합니다.
    $$ E(x*{t,k}) = - \frac{1}{H_2 W_2} \sum*{h,w} \sum*{c} p*{h,w,c}^{t,k} \log(p\_{h,w,c}^{t,k}) $$
  - 엔트로피 순위(entropy-ranking)를 통해 배치 내 같은 위치의 패치들을 엔트로피가 낮은 '쉬운(easy)' 그룹 $I_{t,k}^{\circ}$과 높은 '어려운(hard)' 그룹 $I_{t,k}^{\bullet}$으로 나눕니다.
  - **적대적 학습 손실 ($L_{ADV}$)**: 판별자 $D$는 쉬운 패치와 어려운 패치를 구별하고, 타겟 모델 $T$는 어려운 패치를 쉬운 패치처럼 보이게 하여 패치 간의 간극을 줄이도록 훈련됩니다.
    $$ L*{ADV}(I*{t}^{\bullet},I*{t}^{\circ}) = - \sum*{k}^{K^{2}} \sum*{d,e}^{B/2} \log(1-D(k,i*{t,k}^{e})) + \log(D(k,i\_{t,k}^{d})) $$
- **최종 타겟 모델 목적 함수**: $L_{T AR} + \alpha L_{MAE} + \beta L_{ss}^{DAD} + \gamma L_{ADV}$를 최소화합니다. 여기서 $L_{TAR}$는 타겟 유사 레이블 기반의 자가 감독 손실(예: MaxSquare 손실)입니다.

## 📊 Results

- **합성-실제 적응 (GTA5 → Cityscapes, SYNTHIA → Cityscapes)**:
  - SFDA는 소스 데이터 접근 없이도 MinEnt, AdaptSegNet 등 기존 소스-기반 UDA 방법들보다 우수한 성능을 보였습니다.
  - IPSM 모듈을 통합한 SFDA는 SOTA 소스-기반 UDA 방법들과 비교하여 경쟁력 있는 mIoU를 달성했으며, 특히 SYNTHIA $\rightarrow$ Cityscapes 시나리오에서는 신호등, 표지판, 오토바이와 같은 작은 객체 분할에서 뛰어난 성능을 보였습니다.
- **교차 도시 적응 (Cityscapes → NTHU)**:
  - 더 작은 도메인 시프트 시나리오에서도 SFDA(IPSM 포함)는 MaxSquare와 같은 최고의 UDA 방법과 경쟁력 있는 성능을 달성했습니다.
  - 'transfer only' 결과는 SFKT만으로도 타겟 도메인에서 유용한 지식을 얻을 수 있음을 보여주었습니다.
- **Ablation Study**:
  - SFKT의 핵심 구성 요소인 BNS 손실과 DAD 손실 모두 성능 향상에 기여하며, 두 손실을 결합했을 때 가장 좋은 결과를 보였습니다. DAD 손실이 일반적으로 BNS 손실보다 효과적이었습니다.
  - 생성된 가짜 샘플의 시각화는 DAD나 BNS 손실이 없으면 작은 객체의 인식이나 원래 소스 도메인의 의미론적 분포 보존에 어려움이 있음을 시사했습니다.
- **하이퍼파라미터 분석**: $\alpha=1.0$, $\beta=0.5$ (및 $\tau=\beta$)가 최적의 성능을 보였고, IPSM의 패치 개수 $K$는 3에서 5 사이가 합리적이었습니다.

## 🧠 Insights & Discussion

SFDA 프레임워크는 소스 데이터에 대한 접근 없이도 의미론적 분할을 위한 도메인 적응 문제를 효과적으로 해결할 수 있음을 보여줍니다. 듀얼 어텐션 증류(DAD) 메커니즘을 통해 소스 모델의 픽셀 수준 문맥 정보를 성공적으로 전달하고, 인트라-도메인 패치 수준 자가 감독(IPSM)을 통해 타겟 도메인 내의 신뢰할 수 있는 정보를 활용함으로써, 소스 데이터 없이도 SOTA 소스-기반 UDA 방법과 필적하는 성능을 달성했습니다. 이는 데이터 프라이버시가 중요하거나 소스 데이터가 비공개인 실제 시나리오에서 매우 실용적인 해결책을 제시합니다. 그러나 현재 접근 방식은 생성적 가짜 샘플 합성의 한계로 인해 고해상도 이미지 분할 작업에는 적합하지 않다는 제약이 있으며, 이는 향후 연구에서 개선될 여지가 있습니다.

## 📌 TL;DR

소스 데이터 없이 의미론적 분할을 위한 UDA(Source-Free UDA) 문제 해결을 위해, 이 논문은 **SFDA** 프레임워크를 제안합니다. SFDA는 소스 모델로부터의 지식 전달(듀얼 어텐션 증류, DAD 활용)과 타겟 도메인 내 패치 수준 자가 감독(IPSM 활용)을 결합합니다. 실험 결과, SFDA는 소스 데이터가 없는 상황에서도 기존 소스-기반 UDA 방법들과 경쟁력 있는 성능을 달성하며, 실제 데이터 프라이버시 문제에 대한 효과적인 해결책을 제시합니다.
