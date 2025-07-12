# MVP-SEG: Multi-View Prompt Learning for Open-Vocabulary Semantic Segmentation
Jie Guo, Qimeng Wang, Yan Gao, Xiaolong Jiang, Xu Tang, Yao Hu, Baochang Zhang

## 🧩 Problem to Solve
*   CLIP(Contrastive Language-Image Pretraining)은 개방형 어휘(open-vocabulary) 제로샷(zero-shot) 이미지 레벨 인식에는 탁월하지만, 픽셀 레벨 작업(예: 의미론적 분할)에 직접 적용할 경우 성능이 저조하거나 불완전한 분할을 유발한다.
*   CLIP은 이미지 전체의 맥락을 학습하므로, 특정 객체에 대한 픽셀 레벨의 세밀한 특징보다는 가장 두드러지는 부분에만 집중하여 불완전한 객체 분할이 이루어지는 경향이 있다.
*   따라서, CLIP의 제로샷 능력을 픽셀 레벨 의미론적 분할에 효과적으로 적용하기 위해 이미지-픽셀(image-to-pixel) 수준의 적응(adaptation) 방식이 필요하다.

## ✨ Key Contributions
*   개방형 어휘 의미론적 분할에서 CLIP의 제로샷 능력을 활용하기 위한 이미지-픽셀 적응의 중요성을 입증하고, 제안하는 `MVP-SEG`가 이러한 적응을 성공적으로 수행하여 뛰어난 성능 향상을 가져옴을 보여줍니다.
*   각 프롬프트가 객체의 다른 부분에 주의를 기울이도록 다중 뷰(multi-view) 학습 가능한 프롬프트를 구축하기 위해 `Orthogonal Constraint Loss (OCLoss)`를 설계하여 협력적으로 정확하고 완전한 분할을 제공합니다. 또한, 클래스별 분할 노이즈를 제거하기 위해 `Global Prompt Refining (GPR)` 모듈을 도입합니다.
*   세 가지 주요 벤치마크에 대한 광범위한 실험을 통해 `MVP-SEG`에 지식 전이(knowledge transfer) 단계를 결합한 `MVP-SEG+`가 모든 벤치마크에서 SOTA(State-Of-The-Art) 성능을 보고하며, PASCAL VOC 및 PASCAL Context 데이터셋에서는 완전 지도(fully-supervised) 방식보다도 우수한 성능을 달성합니다.

## 📎 Related Works
*   **Vision-language Models (VL Models):** CLIP, ALIGN, CoCa, Beit-V3와 같이 대규모 이미지-텍스트 쌍으로 사전 훈련된 모델들은 이미지-텍스트 특징을 정렬하여 개방형 세계 지식을 시각적 및 텍스트 피처에 주입하며, 제로샷 시각 인식 및 생성에 활용됩니다.
*   **Pre-Training and Self-Training:** 대규모 데이터셋(ImageNet, JFT 등)으로 모델을 사전 훈련하고, 다운스트림 작업에 미세 조정 또는 프롬프트 학습을 통해 적용합니다. 자가 훈련(self-training)은 지도 모델이 비레이블 데이터에 대해 의사 레이블(pseudo labels)을 생성하고, 이를 통해 모델을 미세 조정하는 준지도(semi-supervised) 방식입니다.
*   **Zero-Shot Segmentation:** 훈련 중 보지 못한 클래스까지 포함하여 픽셀 레벨 분류를 수행하는 연구 분야입니다. SPNet, ZS3Net 등이 대표적이며, CLIP은 텍스트-시각 피처 공간을 연결하는 데 활용됩니다(MaskCLIP 등).
*   **Prompt Learning:** NLP에서 유래한 개념으로, `"[CLS]의 사진"`과 같은 텍스트 템플릿을 사용하여 VL 사전 훈련 모델을 다운스트림 작업에 적용합니다. CoOp은 수동 프롬프트 템플릿의 한계를 극복하기 위해 연속적인 학습 가능한 프롬프트를 도입했으며, 본 논문은 이를 다중 뷰 패러다임으로 확장합니다.

## 🛠️ Methodology
`MVP-SEG`는 학습 가능한 다중 프롬프트를 통해 CLIP 피처를 이미지-픽셀 레벨로 적응시키는 핵심 방법론을 제공하며, `MVP-SEG+`는 여기에 지식 전이 단계를 추가합니다.

*   **`MVP-SEG` (Multi-View Prompt Learning for Segmentation):**
    *   **비전 인코더:** 고정된 CLIP 이미지 인코더를 수정하여 최종 이미지 특징 맵 $F$를 얻습니다. 구체적으로, CLIP의 `AttnPool` 레이어에서 쿼리(`Proj_{q}`) 및 키(`Proj_{k}`) 임베딩 레이어를 제거하고, $1 \times 1$ 컨볼루션 레이어로 구현된 `Proj_{v}`와 `Proj_{c}`를 특징 맵 $X$에 적용하여 $F = \text{Proj}_{c}(\text{Proj}_{v}(X))$를 생성합니다.
    *   **텍스트 인코더 및 다중 뷰 프롬프트:** 고정된 CLIP 텍스트 인코더를 사용합니다. $k+1$개의 학습 가능한 프롬프트 $P = \{p_{0}, p_{1}, ..., p_{k}\}$를 초기화하며, $p_{0}$는 전역 분류 프롬프트, $p_{1}, ..., p_{k}$는 $k$개의 분할 프롬프트입니다.
    *   각 프롬프트 $p_{i}$를 클래스 이름 `cls_{c}`와 결합하여 문장 $s^{c}_{i} = \text{CONCAT}(p_{i}, W(\text{cls}_{c}))$을 구성하고, CLIP 텍스트 인코더에 입력하여 텍스트 표현 벡터 $t^{c}_{i}$를 얻습니다.
    *   각 $t^{c}_{i}$는 특징 맵 $F$에 대한 픽셀 레벨 분류를 수행하여 마스크 맵 $m^{c}_{i}$를 생성합니다 ($m^{c}_{0}$는 전역 분류 마스크, $m^{c}_{1}, ..., m^{c}_{k}$는 분할 마스크).
    *   최종 분할 결과 $m_{f}$는 모든 분할 마스크를 합산하여 계산됩니다:
        $$m_{f} = \text{softmax}\left(\tau_{1} \sum_{i=1}^{k} m_{i}\right)$$
    *   **직교 제약 손실 (Orthogonal Constraint Loss, OCLoss):** 각 프롬프트가 다른 객체 부분에 주의를 기울이도록 프롬프트 학습을 감독하기 위해 도입됩니다. 각 분할 프롬프트 $p_{i}$의 평균 벡터 $p'_{i}$를 구한 후, 다음과 같이 계산됩니다:
        $$L_{OC} = \sum_{i=1}^{k} \sum_{j=i+1}^{k} \frac{|p'_{i} \cdot p'_{j}|}{\|p'_{i}\| \|p'_{j}\|}$$
*   **전역 프롬프트 정제 (Global Prompt Refining, GPR):**
    *   CLIP의 강력한 이미지 분류 능력을 분할에 통합하기 위해 전역 분류 프롬프트(`p_{0}`)에서 얻은 이미지 분류 점수 $g_{c}$를 사용하여 분할 마스크에서 클래스별 노이즈를 제거합니다:
        $$g_{c} = \text{sigmoid}\left(\frac{m^{c}_{0} \cdot \text{softmax}(\gamma m^{c}_{0})}{\tau_{2}}\right)$$
    *   전역 분류 손실 $L_{cls}$는 $L_{cls} = -\sum_{c} y_{c} \log(g_{c})$입니다.
    *   최종 분할 마스크 $m_{c}$는 융합된 분할 마스크 $m_{f}$에 전역 분류 점수 $g_{c}$를 곱하여 얻습니다: $m_{c} = m^{f}_{c} g_{c}$.
    *   분할 손실 $L_{seg}$는 $L_{seg} = -\sum_{H} \sum_{W} \sum_{c=1}^{C} m^{*}_{c} \log(m_{c})$입니다.
    *   전체 손실 함수는 다음과 같습니다:
        $$L = \lambda_{1}L_{seg} + \lambda_{2}L_{cls} + \lambda_{3}L_{OC}$$
    *   프롬프트 학습 단계에서는 이미지 인코더와 텍스트 인코더는 고정되고, 프롬프트 벡터 $P$ 및 스케일 $\tau_{1}$, $\tau_{2}$만 학습됩니다.

*   **`MVP-SEG+`:**
    *   `MVP-SEG`와 지식 전이(knowledge transfer) 단계로 구성된 전체 프레임워크입니다.
    *   **지식 전이:** `MVP-SEG`에서 학습된 프롬프트는 의사 레이블(pseudo labels)을 생성하는 데 사용되며, 이 의사 레이블을 통해 CLIP의 제로샷 지식을 segmentation network(`DeepLabv2`, `DeepLabv3+` 등)로 전이합니다.
    *   이 단계는 의사 레이블 훈련(pseudo-label training)과 자가 훈련(self-training)의 두 단계로 구성됩니다. 분할 모델의 분류기는 `MVP-SEG`의 다중 뷰 프롬프트로 대체되어 개방형 어휘 능력을 유지합니다.

## 📊 Results
*   **`MVP-SEG` Ablation Studies:**
    *   **MaskCLIP Baseline 대비 개선:** `MVP-SEG`는 MaskCLIP 대비 COCO Stuff의 unseen 클래스 mIoU를 8.4%p, PASCAL VOC에서는 12.7%p 향상시켜, 다중 뷰 학습 가능한 프롬프트가 픽셀 레벨 작업에서 CLIP 성능을 크게 개선함을 입증했습니다.
    *   **학습 가능한 프롬프트의 효과:** 85개의 수동 프롬프트를 1개의 학습 가능한 프롬프트로 대체하는 것만으로도 COCO Stuff에서 unseen 클래스 성능이 3.6%p 향상되어 학습 가능한 프롬프트의 효율성을 보여주었습니다.
    *   **프롬프트 수:** COCO Stuff에서 프롬프트 수가 증가할수록 성능이 향상되며, 3개의 프롬프트에서 최고 성능에 도달했습니다.
    *   **`OCLoss`의 영향:** `OCLoss`를 적용했을 때 COCO Stuff의 unseen 클래스 mIoU가 16.4%에서 19.9%로 증가하여, 다중 뷰 프롬프트 학습 방법의 효과를 확인했습니다.
    *   **`GPR`의 영향:** `GPR` 모듈을 추가했을 때 COCO Stuff hIoU가 17.9%에서 19.2%로, PASCAL VOC에서는 48.3%에서 53.1%로 전반적인 성능이 향상되었습니다. 이는 `GPR`이 오탐(false positive) 클래스의 마스크를 효과적으로 필터링함을 시사합니다.
    *   **프롬프트의 일반화 능력:** 한 데이터셋에서 학습된 프롬프트가 다른 데이터셋에서도 baseline을 크게 능가하는 성능을 보여 뛰어난 일반화 능력을 입증했습니다.
*   **SOTA 비교 (`MVP-SEG+`):**
    *   `MVP-SEG+`는 COCO Stuff, Pascal VOC, Pascal Context에서 이전 SOTA 방법 대비 unseen 클래스 mIoU를 각각 1.1%p, 1.3%p, 0.8%p 향상시켰습니다.
    *   모든 데이터셋에서 hIoU 기준 이전 SOTA를 일관되게 능가했습니다 (각각 +0.3%p, +0.5%p, +0.6%p).
    *   `MVP-SEG+`의 성능은 완전 지도 방식과 비견할 만하거나, PASCAL VOC 및 PASCAL Context 데이터셋에서는 심지어 능가하는 결과를 보여주었습니다. 이는 CLIP을 픽셀 레벨 작업에 효과적으로 적응시켰음을 의미합니다.
*   **정성적 결과:** MaskCLIP+ 대비 더 완전한 마스크를 얻었고, 오탐 클래스를 필터링할 수 있었습니다. 웹에서 수집한 희귀 객체 클래스(예: 아이언맨, 캡틴 아메리카)에 대해서도 정확한 분할을 보여 개방형 어휘 능력을 입증했습니다.

## 🧠 Insights & Discussion
*   본 연구는 CLIP의 이미지 레벨 지식을 픽셀 레벨 분할 작업에 성공적으로 적응시키는 것이 매우 중요함을 명확히 보여줍니다. 단순히 CLIP 피처를 사용하는 것을 넘어, 작업 특성에 맞게 프롬프트를 학습시키는 것이 성능 향상에 필수적입니다.
*   객체 부분이 명확하지 않은 개방형 어휘 환경에서, 다중 뷰 학습 가능한 프롬프트가 객체의 다양한 부분에 집중하여 분할 결과의 완전성과 정확성을 높이는 "부분 기반 표현"의 효과를 효과적으로 입증합니다. `OCLoss`는 이러한 프롬프트들이 상호 직교적으로 학습되도록 유도하여 각 프롬프트가 고유한 특징을 포착하도록 돕습니다.
*   `GPR` 모듈은 이미지 레벨 분류 능력을 활용하여 픽셀 레벨 분할 결과의 노이즈를 제거함으로써, 전역적인 맥락 정보가 픽셀 레벨 예측을 정제하는 데 기여할 수 있음을 보여줍니다.
*   `MVP-SEG+`가 완전 지도 방식보다 우수한 성능을 달성한 것은 사전 훈련된 대규모 시각-언어 모델(CLIP)의 잠재력을 최대한 끌어내어 제한된 레이블 데이터만으로도 고성능을 달성할 수 있음을 시사하며, 이는 개방형 어휘 설정에서 실용적인 의미를 갖습니다.

## 📌 TL;DR
`MVP-SEG`는 CLIP의 이미지 레벨 편향으로 인한 개방형 어휘 의미론적 분할의 불완전성 문제를 해결하기 위해, 객체의 다양한 부분에 집중하는 다중 뷰 학습 가능한 프롬프트와 `OCLoss`, 그리고 클래스별 노이즈 제거를 위한 `GPR`을 제안합니다. 이를 통해 CLIP을 픽셀 레벨 작업에 효과적으로 적응시키며, `MVP-SEG+`는 SOTA 성능을 달성하고 PASCAL VOC 및 PASCAL Context 데이터셋에서는 완전 지도 방식보다도 우수한 결과를 보여줍니다.