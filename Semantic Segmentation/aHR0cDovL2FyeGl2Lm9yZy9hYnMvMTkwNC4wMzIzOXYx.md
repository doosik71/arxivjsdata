# ShapeMask: Learning to Segment Novel Objects by Refining Shape Priors

Weicheng Kuo, Anelia Angelova, Jitendra Malik, Tsung-Yi Lin

## 🧩 Problem to Solve

기존의 인스턴스 분할(Instance Segmentation) 방법들은 새로운 범주의 객체에 대한 픽셀 단위 마스크 주석(pixel-wise mask annotations)에 크게 의존합니다. 하지만 이러한 마스크 주석을 대규모로 확보하는 것은 비용이 많이 들고 어렵기 때문에, 모델이 학습되지 않은 새로운 범주에 대해 일반화(generalization)하는 데 한계가 있습니다. 이는 인스턴스 분할의 실제 적용 가능성을 제한하는 주요 문제입니다.

## ✨ Key Contributions

* **새로운 일반화 방법론 제안:** 객체 모양(shape priors) 및 인스턴스 임베딩(instance embeddings)이라는 중간 개념을 학습하여 새로운 범주의 객체를 분할하는 ShapeMask를 제안합니다.
* **최고 수준의 일반화 성능:** 범주 간 학습(cross-category learning) 설정에서 기존 최첨단 방법(Mask X R-CNN)보다 최대 **9.4 AP** 높은 성능을 달성하여, 마스크 주석이 없는 범주에 대한 탁월한 일반화 능력을 입증했습니다.
* **효율성 및 견고성:** 11시간 이내에 효율적인 훈련(기존 Mask R-CNN 대비 4배 빠름)을 완료하고, 이미지당 150ms의 빠른 추론 시간을 제공합니다. 또한, 부정확한 탐지(inaccurate detections) 및 적은 학습 데이터에 대해서도 높은 견고성을 보입니다.
* **경량 마스크 브랜치:** 마스크 브랜치 용량을 130배 줄이고도 경쟁력 있는 성능을 유지하며 6배 빠르게 실행될 수 있음을 보여줍니다.
* **로봇 공학 분야 적용 가능성:** COCO 데이터셋으로만 학습했음에도 불구하고, 로봇 공학 환경의 새로운 객체들을 성공적으로 분할하여 실제 적용 가능성을 입증했습니다.

## 📎 Related Works

* **인스턴스 분할 접근법:**
  * **탐지 기반(Detection-based):** Mask R-CNN, FCIS, MaskLab, Faster R-CNN 등. 바운딩 박스를 먼저 탐지하고 해당 영역 내에서 마스크를 예측합니다. COCO 및 Cityscapes에서 최첨단 성능을 보였습니다.
  * **그룹화 기반(Grouping-based):** Deep Watershed Transform, InstanceCut, MCG 등. 픽셀 단위 예측(방향 벡터, 인접도 등)을 통해 객체 인스턴스를 그룹화합니다. 종종 시맨틱 분할(semantic segmentation)에 의존하여 클래스 정보를 얻습니다.
* **모양 사전(Shape Priors):** 객체 분할에서 모양 정보를 활용하는 것은 기존 연구에서도 사용되었습니다 (예: 확률 프레임워크의 유니라리(unaries), 제안 영역 강화, 상향식 그룹화 지원).
* **약한/부분적으로 지도된 인스턴스 분할(Weakly/Partially Supervised Instance Segmentation):**
  * GrabCut 같은 그룹화 기반 알고리즘을 활용하여 의사 마스크 레이블(pseudo mask labels)을 얻는 방법 [23].
  * 경계 탐지기(boundary detector)와 그룹화를 사용하는 오픈셋(open-set) 인스턴스 분할 [36].
  * 이미지 수준 감독(image-level supervision)으로부터 학습하는 방법 [47].
  * **Mask X R-CNN [20]:** 바운딩 박스 주석만 있는 범주에 대한 부분적으로 지도된 인스턴스 분할 문제를 다루며, ShapeMask의 주요 비교 대상입니다.

## 🛠️ Methodology

ShapeMask는 바운딩 박스 탐지 결과로부터 정확한 인스턴스 마스크를 점진적으로 정제하는 다단계 학습 프레임워크를 제안합니다.

1. **Shape Recognition (모양 인식):**
    * **모양 사전($H$):** 학습 세트의 마스크 주석들을 $K$-평균 군집화하여 정규화된 "모양 베이스" 또는 "모양 사전" $H = \{S_1, S_2, ..., S_K\}$를 얻습니다. 이는 객체와 같은 예측을 위한 강력한 단서를 제공합니다. 클래스별 설정에서는 $C \times K$개, 클래스 비의존적 설정에서는 $K$개의 모양 사전이 생성됩니다.
    * **모양 추정:**
        * 주어진 바운딩 박스 $B$ 내의 특징 맵 $X$에서 인스턴스 특징 임베딩 $x_{\text{box}}$를 추출합니다: $x_{\text{box}} = \frac{1}{|B|} \sum_{(i,j) \in B} X_{(i,j)}$ (식 1).
        * $x_{\text{box}}$는 모양 사전 $S_k$들을 조합하기 위한 가중치 $w_k$를 예측하는 데 사용됩니다: $w_k = \text{softmax}(\phi_k(x_{\text{box}}))$ (식 2).
        * 예측된 모양 $S = \sum_{k=1}^{K} w_k S_k$는 바운딩 박스 $B$에 맞게 크기 조절되어 "탐지 사전(detection prior)" $S_{\text{prior}}$를 생성합니다.
        * 모양 추정 단계는 ground-truth 마스크 $S_{\text{gt}}$에 대한 픽셀 단위 평균 제곱 오차(MSE) 손실로 학습됩니다: $L_{\text{prior}} = \text{MSE}(S_{\text{prior}}, S_{\text{gt}})$ (식 3).
2. **Coarse Mask Prediction (거친 마스크 예측):**
    * 이전 단계의 $S_{\text{prior}}$는 1x1 컨볼루션 레이어 $g$를 통해 이미지 특징 $X$와 동일한 차원으로 임베딩된 후 더해져, 사전 조건부 특징 맵 $X_{\text{prior}}$를 생성합니다: $X_{\text{prior}} = X + g(S_{\text{prior}})$ (식 4).
    * 4개의 컨볼루션 레이어로 구성된 함수 $f$가 $X_{\text{prior}}$로부터 거친 인스턴스 마스크 $S_{\text{coarse}}$를 디코딩합니다: $S_{\text{coarse}} = f(X_{\text{prior}})$ (식 5).
    * $S_{\text{coarse}}$는 픽셀 단위 교차 엔트로피(CE) 손실로 학습됩니다: $L_{\text{coarse}} = \text{CE}(S_{\text{coarse}}, S_{\text{gt}})$ (식 6).
3. **Shape Refinement by Instance Embedding (인스턴스 임베딩을 통한 모양 정제):**
    * 이진화된 거친 마스크 $S_{\text{coarse}}$ 내의 $X_{\text{prior}}$ 특징을 평균 풀링하여 인스턴스 마스크 임베딩 $x_{\text{mask}}$를 얻습니다: $x_{\text{mask}} = \frac{1}{|S_{\text{coarse}}|} \sum_{(i,j) \in S_{\text{coarse}}} X_{\text{prior}(i,j)}$ (식 7).
    * 이미지 특징 $X_{\text{prior}}$에서 $x_{\text{mask}}$를 빼서 인스턴스 중심 특징(instance-centered features) $X_{\text{inst}}$를 생성합니다: $X_{\text{inst}(i,j)} = X_{\text{prior}(i,j)} - x_{\text{mask}}$ (식 8). 이는 모델이 객체 인스턴스를 나타내는 저차원 특징을 학습하도록 유도합니다.
    * 마스크 디코딩 브랜치(추가적인 업샘플링 레이어 포함)를 통해 최종적인 세밀한 마스크 $S_{\text{fine}}$을 예측합니다.
    * $S_{\text{fine}}$은 더 높은 해상도의 ground-truth 마스크 $S_{\text{gt}}$에 대한 픽셀 단위 CE 손실로 학습됩니다: $L_{\text{fine}} = \text{CE}(S_{\text{fine}}, S_{\text{gt}})$ (식 9).

* **클래스 비의존적 학습을 통한 일반화:** 새로운 범주에 대한 일반화를 위해 ShapeMask는 클래스 비의존적 학습을 채택합니다. 모든 범주의 인스턴스 마스크를 통합하여 더 많은 수의 모양 사전($K$)을 생성하여 다양한 모양을 포착합니다.
* **구현 세부 사항:**
  * 바운딩 박스 탐지를 위해 RetinaNet 원스테이지 탐지기(one-stage detector)를 사용합니다.
  * 추론 시 탐지의 불완전함을 모방하기 위해 Gaussian 노이즈를 추가한 **jittered groundtruths**로 학습하여 모델의 견고성을 높입니다.
  * ROIAlign 대신 간단한 크롭을 수행하고, FPN을 활용하여 하드웨어 가속기(TPU)에서 효율적인 학습을 가능하게 합니다.

## 📊 Results

* **새로운 범주에 대한 일반화 (부분적으로 지도된 설정):**
  * COCO 데이터셋에서 ResNet-101-FPN 백본 사용 시, 기존 Mask X R-CNN보다 VOC$\rightarrow$Non-VOC 전이에서 **6.4 AP**, Non-VOC$\rightarrow$VOC 전이에서 **3.8 AP** 향상된 성능을 보였습니다.
  * NAS-FPN과 같은 더 강력한 백본 사용 시, Mask X R-CNN 대비 각각 **9.4 AP** 및 **6.2 AP**까지 격차를 벌렸습니다.
  * Mask X R-CNN보다 Oracle 상한과의 차이가 훨씬 작습니다.
  * **적은 데이터로의 일반화:** 전체 학습 데이터의 1/1000만 사용했을 때도 Mask X R-CNN (전체 데이터로 학습)보다 우수한 성능을 보였습니다. 1%의 학습 마스크 데이터만으로도 Mask X R-CNN보다 **2.0 AP** 높은 성능을 달성했습니다.
  * **로봇 공학 데이터셋 일반화:** COCO로만 학습되었음에도, 로봇 공학 데이터셋의 봉제 인형, 문서, 휴지 상자 등 COCO에 없는 새로운 객체를 성공적으로 분할했습니다.
* **완전히 지도된 인스턴스 분할:**
  * ResNet-101-FPN 백본으로 Mask R-CNN보다 **1.7 AP** 높은 성능을 달성했습니다.
  * 더 강력한 백본(ResNet-101-NAS-FPN)으로 Mask R-CNN 및 MaskLab보다 각각 **2.9 AP** 및 **2.7 AP** 높은 성능을 보였으며, PANet에는 2.0 AP 차이로 뒤쳐졌습니다 (특화된 기법 없이).
  * **효율성:** TPU에서 11시간 만에 훈련을 완료하여 Mask R-CNN보다 4배 빠릅니다. 추론 시간은 이미지당 150-200ms입니다.
* **부정확한 탐지에 대한 견고성:** 다운사이징되거나 교란된 바운딩 박스 탐지에도 Mask R-CNN보다 **5.3 AP** 높은 견고성을 보였습니다. 학습 시 지터링(jittering)을 추가하면 더욱 견고해집니다.
* **모델 어블레이션:** 모양 사전과 인스턴스 임베딩 모두 성능 향상에 크게 기여함을 확인했습니다 (부분적으로 지도된 설정에서 각각 약 12 AP 및 5 AP 향상).
* **경량 마스크 브랜치:** 마스크 브랜치 채널 수를 16개로 줄여도 35.8 AP를 달성하며 Mask R-CNN보다 0.4 AP 높고, 파라미터는 130배, FLOPs는 23배 적게 사용하면서 4.6ms로 빠르게 작동합니다.

## 🧠 Insights & Discussion

* ShapeMask의 핵심은 **중간 표현(모양 사전)**과 **인스턴스 특정 학습(인스턴스 임베딩)**을 통해 새로운 객체 범주에 대한 일반화 능력을 크게 향상시켰다는 점입니다. 모양 사전은 모델의 출력 공간을 정규화하여 불가능한 모양 예측을 방지하고, 객체들이 공유하는 기본적인 형태 정보를 활용합니다. 인스턴스 임베딩은 인스턴스 고유의 시각적 특징을 포착하여, 모델이 이전에 보지 못한 객체라도 같은 인스턴스에 속하는 픽셀들을 효과적으로 그룹화할 수 있게 합니다.
* 효율적인 설계(원스테이지 탐지기, 지터링된 ground-truth, TPU 최적화) 덕분에 빠른 학습 및 추론 속도를 달성하면서도 성능 저하가 거의 없습니다. 이는 실제 애플리케이션에 매우 중요합니다.
* 불정확한 탐지 결과에 대한 견고성은 ShapeMask의 실용성을 높여줍니다. 실제 환경에서는 완벽한 탐지가 어렵기 때문에, 이러한 견고성은 모델의 강점입니다.
* 로봇 공학과 같은 완전히 새로운 환경의 객체에 대한 놀라운 일반화 능력은 ShapeMask가 "미지의 인스턴스 분할" 문제를 해결하는 데 중요한 진전을 이루었음을 시사합니다.
* 제한점으로는, 최첨단 특화된 기법들을 적용하지 않았을 때는 완전히 지도된 설정에서 PANet과 같은 방법에 비해 약간 뒤처질 수 있습니다.

## 📌 TL;DR

**문제:** 기존 인스턴스 분할은 새로운 객체 범주에 대한 픽셀 단위 주석 의존도가 높아 일반화 능력이 부족합니다.
**해결책:** ShapeMask는 바운딩 박스를 **모양 사전**으로 점진적으로 정제하고 **인스턴스 임베딩**을 학습하여, 새로운 객체 범주를 효과적으로 분할합니다.
**성과:** ShapeMask는 기존 최첨단 방법보다 새로운 범주에 대해 최대 **9.4 AP** 더 높은 성능을 달성했으며, 부정확한 탐지에 견고하고, 학습 및 추론이 매우 효율적입니다.
