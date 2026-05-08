# Demonstrating and Reducing Shortcuts in Vision-Language Representation Learning

Maurits Bleeker, Mariya Hendriksen, Andrew Yates, Maarten de Rijke (2024)

## 🧩 Problem to Solve

본 논문은 시각-언어 표현 학습(Vision-Language Representation Learning), 특히 하나의 이미지에 여러 개의 캡션이 대응되는 상황에서 Contrastive Learning(CL)이 가질 수 있는 근본적인 한계인 **Shortcut Learning** 문제를 다룬다.

일반적으로 Vision-Language 모델(VLM)은 이미지와 텍스트 간의 정렬을 최대화하는 Contrastive Loss를 사용하여 학습한다. 이때 각 캡션은 이미지의 공통된 정보(Shared information)뿐만 아니라, 특정 캡션만이 담고 있는 고유한 정보(Unique/Caption-specific information)를 동시에 포함한다. 연구진은 Contrastive Loss가 이러한 모든 태스크 관련 정보를 통합하여 **Task-optimal representation**을 학습하는지, 아니면 단순히 손실 함수를 빠르게 최소화할 수 있는 쉬운 특징인 **Shortcut**만을 학습하는지 분석하고자 한다.

이 문제의 중요성은 모델이 Shortcut에 의존할 경우, 훈련 데이터셋의 통계적 편향이나 단순한 패턴에만 최적화되어 실제 환경에서의 일반화 성능이 떨어지고, 이미지의 세밀한 세부 사항을 무시하는 결과로 이어지기 때문이다. 따라서 본 논문의 목표는 Contrastive VL 학습에서 Shortcut Learning이 발생하는지 제어된 환경에서 증명하고, 이를 완화할 수 있는 방법을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같이 요약할 수 있다.

1. **SVL(Synthetic Shortcuts for Vision-Language) 프레임워크 제안**: 이미지와 텍스트 쌍에 인위적인 식별자(Synthetic Shortcut)를 주입하여, 모델이 의미론적 정보 대신 단순한 식별자에 의존하는지 정량적으로 측정할 수 있는 통제된 실험 환경을 구축하였다.
2. **Contrastive Learning의 하위 최적성(Suboptimality) 증명**: 수학적 분석(Theorem 1)과 실험을 통해, InfoNCE와 같은 Contrastive Loss가 Task-optimal representation이 아닌 **Minimally sufficient representation**만을 학습하려는 경향이 있음을 보였다. 이는 모델이 가장 배우기 쉬운 최소한의 공유 특징만을 포착하고 나머지 유용한 정보를 억제(Suppression)함을 의미한다.
3. **Shortcut 완화 전략 분석**: Latent Target Decoding(LTD)과 Implicit Feature Modification(IFM)이라는 두 가지 기법을 적용하여 Shortcut Learning을 어느 정도 줄일 수 있는지 검증하였으며, 특히 LTD가 모델의 붕괴를 막는 데 더 효과적임을 확인하였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 바탕으로 차별점을 둔다.

- **Multi-view Representation Learning**: 기존의 다중 뷰 학습은 각 뷰가 동일한 태스크 관련 정보를 공유한다는 가정하에 상호 정보량(Mutual Information, MI)을 최대화한다. 그러나 본 논문은 VL 데이터셋에서 각 캡션(뷰)이 서로 다른 고유 정보를 가질 수 있다는 점에 주목하여, 단순히 MI를 최대화하는 것이 최적의 표현을 학습하는 것과 일치하지 않음을 지적한다.
- **Shortcut Learning**: Geirhos et al. (2020) 등이 정의한 Shortcut Learning은 벤치마크에서는 성능이 좋지만 실제 환경에서는 실패하는 결정 규칙을 학습하는 현상이다. 본 논문은 이를 VL 도메인으로 확장하여, 이미지-텍스트 정렬 과정에서 발생하는 특징 억제(Feature Suppression) 현상을 분석한다.
- **Contrastive VL Models**: CLIP과 같은 대규모 모델과 VSE++와 같은 소규모 모델의 구조적 차이에도 불구하고, 두 모델 모두 Contrastive Loss의 특성상 Shortcut에 취약하다는 점을 실험적으로 입증하며 기존 모델들의 학습 메커니즘에 대한 비판적 시각을 제공한다.

## 🛠️ Methodology

### 1. Theoretical Analysis

논문은 InfoMax 최적화 목적 함수가 이미지 표현 $z_I$와 캡션 표현 $z_C$ 사이의 상호 정보량 $I(z_I; z_C)$를 최대화하는 것과 같다고 설명한다. 이때 **Minimally Sufficient Image Representation**($z^{MIN}_{I \to C}$)은 캡션 $x_C$와 공유되는 최소한의 정보만을 포함하며, 공유되지 않는 정보는 억제한다. 반면 **Task-optimal Image Representation**($z^{OPT}_{I \to K}$)은 모든 매칭 캡션 집합 $K$에 대해 충분한 정보를 모두 포함해야 한다.

본 논문의 **Theorem 1**은 하나의 이미지에 여러 캡션이 있을 때, Contrastive Learning을 통해 학습된 표현은 Minimally Sufficient-할 뿐, 결코 Task-optimal 할 수 없음을 수학적으로 증명한다. 즉, 한 캡션에 최적화되는 과정에서 다른 캡션만이 가진 고유 정보가 억제되기 때문이다.

### 2. SVL (Synthetic Shortcuts for Vision-Language) Framework

모델이 Shortcut에 의존하는지 확인하기 위해 다음과 같은 인위적 Shortcut을 주입한다.

- **이미지 측면**: 원본 이미지 상단에 MNIST 숫자 이미지를 오버레이(Overlay)한다.
- **텍스트 측면**: 캡션 끝에 해당 숫자들을 텍스트 토큰으로 추가한다.
- 이 Shortcut은 의미론적 내용은 없으나 이미지와 텍스트를 매우 쉽게 매칭시킬 수 있는 강력한 단서가 된다.

### 3. Shortcut 완화 방법론

#### (1) Latent Target Decoding (LTD)

LTD는 Contrastive Loss에 재구성 손실(Reconstruction Loss) $\mathcal{L}_{recon}$을 추가하여 특징 억제를 방지한다.

- **절차**: 캡션의 잠재 표현 $z_C$로부터 원래 캡션을 복원하도록 학습시킨다. 이때 단순한 토큰 복원이 아니라, Sentence-BERT의 잠재 공간으로 매핑하여 그 거리(Distance)를 최소화하는 비-자기회귀(non-auto-regressive) 방식을 사용한다.
- **목표**: Sentence-BERT의 표현이 캡션의 모든 태스크 관련 정보를 담고 있다고 가정하므로, 이를 복원하게 함으로써 인코더가 정보를 임의로 억제하거나 Shortcut에만 의존하는 것을 막는다.
- **수식**: $\mathcal{L}_{InfoNCE+LTD} = \mathcal{L}_{InfoNCE} + \beta \mathcal{L}_{recon}$ (또는 라그랑주 승수를 이용한 제약 조건 형태로 구현)

#### (2) Implicit Feature Modification (IFM)

IFM은 학습 과정에서 모델이 현재 의존하고 있는 특징을 인위적으로 방해하여 더 다양한 특징을 찾게 만드는 방법이다.

- **절차**: Logit 값에 섭동(Perturbation) 예산 $\epsilon$을 적용한다. 긍정 쌍(Positive pair)의 유사도에는 $\epsilon/\tau$를 빼고, 부정 쌍(Negative pair)의 유사도에는 $\epsilon/\tau$를 더한다.
- **목표**: 모델이 현재 쉽게 찾은 Shortcut 특징으로 정답을 맞히려는 시도를 어렵게 만들어, 다른 유용한 특징을 학습하도록 유도한다.
- **수식**:
$$\mathcal{L}^{t2i}_{IFM} = \frac{1}{|B|} \sum_{i \in B} \log \frac{\exp((z^I_i \cdot z^C_i - \epsilon)/\tau)}{\exp((z^I_i \cdot z^C_i - \epsilon)/\tau) + \sum_{j \neq i} \exp((z^I_j \cdot z^C_i + \epsilon)/\tau)}$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Flickr30k, MS-COCO (이미지당 5개 캡션)
- **모델**: CLIP (ResNet-50, Fine-tuning), VSE++ (ResNet-152, From scratch)
- **지표**: Recall sum (R@1, R@5, R@10의 i2t 및 t2i 합산)

### 2. 주요 결과

- **Shortcut에 대한 의존성**:
  - Unique Shortcut을 주입하여 학습했을 때, 평가 시 Shortcut을 포함하면 완벽한 점수를 얻지만, **Shortcut을 제거하면 성능이 급격히 하락**한다.
  - 특히 VSE++는 성능이 0에 수렴하며, CLIP은 Zero-shot 성능보다 낮아진다. 이는 모델이 의미론적 특징을 전혀 학습하지 않고 오직 Shortcut만을 학습했음을 의미한다.
- **Shortcut 비트 수($N$)의 영향**: Shortcut으로 사용하는 숫자의 범위($2^n$)가 넓어질수록(즉, Shortcut이 더 고유해질수록) 모델은 원래의 태스크 관련 정보를 더 많이 무시하고 Shortcut에 더 강하게 의존한다.
- **완화 기법의 효과**:
  - **LTD**: CLIP과 VSE++ 모두에서 Shortcut 학습을 유의미하게 줄였다. 특히 VSE++에서 Shortcut 존재 시 성능 붕괴를 막는 데 효과적이었다.
  - **IFM**: CLIP의 Fine-tuning 시에는 성능 하락을 어느 정도 방지했으나, VSE++처럼 처음부터 학습하는 모델에서는 Shortcut 붕괴를 막지 못했다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 단순한 성능 향상이 아니라, Contrastive Learning의 목적 함수가 가진 이론적 취약점을 SVL이라는 통제된 프레임워크를 통해 명확히 입증하였다. 특히 대규모 사전 학습 모델인 CLIP조차 Fine-tuning 과정에서 기존의 풍부한 특징을 버리고 단순한 Shortcut에 매몰될 수 있다는 점을 밝혀낸 것이 매우 인상적이다.

### 한계 및 비판적 해석

- **부분적 해결**: LTD와 IFM이 성능을 개선시키긴 했으나, Shortcut이 없는 Baseline 모델의 성능 수준까지 회복시키지는 못했다. 이는 현재의 완화 기법들이 Shortcut Learning의 근본적인 해결책이 되기에는 부족하며, 여전히 정보 손실이 발생하고 있음을 시사한다.
- **가정의 단순함**: MNIST 숫자를 이용한 Shortcut은 매우 단순한 형태이다. 실제 데이터셋에 존재하는 더 교묘하고 복잡한 형태의 Shortcut(예: 배경의 특정 색상, 텍스트의 특정 패턴)에 대해서도 동일한 현상이 발생하는지, 그리고 제시된 방법론이 작동할지는 추가 검증이 필요하다.

## 📌 TL;DR

본 연구는 Vision-Language 모델의 Contrastive Learning이 모든 유용한 정보를 학습하는 것이 아니라, 손실을 빠르게 줄일 수 있는 **최소한의 쉬운 특징(Shortcut)**만을 학습하고 나머지 정보를 억제하는 경향이 있음을 증명하였다. 이를 위해 인위적 Shortcut을 주입하는 **SVL 프레임워크**를 제안하였으며, 실험을 통해 CLIP과 VSE++ 모두 이 현상에 취약함을 보였다. 해결책으로 제시된 **LTD(재구성 손실 추가)**와 **IFM(로짓 섭동)** 중 LTD가 더 강력한 억제 방지 효과를 보였으나, 여전히 완전한 해결은 어려웠다. 이 연구는 향후 VLM 학습 시 단순한 Contrastive Loss를 넘어 Task-optimal한 표현을 학습시키기 위한 새로운 목적 함수 설계의 필요성을 제기한다.
