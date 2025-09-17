# Segmentation Transformer: Object-Contextual Representations for Semantic Segmentation

Yuhui Yuan, Xiaokang Chen, Xilin Chen, and Jingdong Wang

## 🧩 Problem to Solve

이 논문은 의미론적 분할(Semantic Segmentation)에서 컨텍스트 정보(Context Information)를 효과적으로 통합하는 문제, 특히 각 픽셀이 속한 객체 클래스의 컨텍스트를 활용하여 픽셀의 표현을 강화하는 방법을 연구합니다. 기존의 다중 스케일 컨텍스트 방법이나 관계형 컨텍스트 방법들이 픽셀과 픽셀 간의 관계에 초점을 맞추는 반면, 이 연구는 픽셀이 속한 '객체'의 컨텍스트를 명시적으로 학습하고 활용함으로써 분할 품질을 향상시키고자 합니다.

## ✨ Key Contributions

- **객체 컨텍스트 표현(Object-Contextual Representations, OCR) 제안:** 픽셀이 속한 객체 클래스의 표현을 활용하여 픽셀을 특징화하는 간단하지만 효과적인 접근 방식을 제시합니다.
- **3단계 OCR 프레임워크:**
  1. 지상 진실(ground-truth) 분할의 감독 하에 소프트 객체 영역(soft object regions)을 학습합니다.
  2. 해당 객체 영역에 속한 픽셀들의 표현을 집계하여 객체 영역 표현을 계산합니다.
  3. 각 픽셀과 각 객체 영역 간의 관계를 계산하고, 모든 객체 영역 표현의 가중치 합으로 각 픽셀의 객체 컨텍스트 표현을 강화합니다.
- **Transformer 인코더-디코더 프레임워크로 OCR 재해석:** 제안된 OCR 스키마가 Transformer 인코더-디코더 구조의 크로스-어텐션(cross-attention) 모듈과 유사함을 보여주며, 이를 Segmentation Transformer로 명명합니다.
- **다양한 벤치마크에서의 우수성 입증:** Cityscapes, ADE20K, LIP, PASCAL-Context, COCO-Stuff 등 여러 의미론적 분할 벤치마크에서 기존 SOTA(State-of-the-Art) 방법을 능가하거나 경쟁력 있는 성능을 달성합니다.
- **효율성 개선:** 기존 다중 스케일 또는 관계형 컨텍스트 방법에 비해 더 적은 파라미터, GPU 메모리, FLOPs, 그리고 더 빠른 추론 시간을 보여줍니다.

## 📎 Related Works

- **다중 스케일 컨텍스트 (Multi-scale context):** ASPP [6], PPM [80], DenseASPP [68] 등 다양한 스케일의 컨텍스트를 캡처하는 방법들. 이들은 주로 공간적 범위에 중점을 둡니다.
- **관계형 컨텍스트 (Relational context):** DANet [18], CFNet [77], OCNet [72, 71] 등 픽셀과 그 주변 컨텍스트 픽셀 간의 관계를 고려하여 표현을 강화하는 Self-Attention [64, 61] 기반 방법들.
- **Double Attention [8] 및 ACFNet [75]:** 픽셀을 일련의 영역으로 그룹화한 다음, 이 영역 표현을 집계하여 픽셀 표현을 강화하는 방법. OCR은 영역 형성 방식과 픽셀-영역 관계 계산 방식에서 이들과 차이를 보입니다.
- **거친-세밀 분할 (Coarse-to-fine segmentation):** 점진적으로 분할 맵을 개선하는 방법들 [17, 20, 34]. OCR은 거친 분할 맵을 직접 사용하는 대신 컨텍스트 표현을 생성하는 데 활용한다는 점에서 다릅니다.
- **영역 기반 분할 (Region-wise segmentation):** 픽셀을 영역으로 조직화하고 각 영역을 분류하는 방법들 [1, 2, 23]. OCR은 영역 자체를 분류하는 대신 영역을 활용하여 픽셀 레이블링을 개선합니다.

## 🛠️ Methodology

제안된 Object-Contextual Representations (OCR) 접근 방식은 다음의 세 가지 주요 단계로 구성됩니다:

1. **소프트 객체 영역 형성 (Soft Object Regions):**

   - 입력 이미지 $I$의 픽셀들을 $K$개의 소프트 객체 영역 $M_{1}, M_{2}, \dots, M_{K}$으로 분할합니다. 각 영역 $M_{k}$는 특정 클래스 $k$에 해당하며, 해당 픽셀이 클래스 $k$에 속하는 정도를 나타내는 2D 맵(또는 거친 분할 맵)으로 표현됩니다.
   - 이 객체 영역들은 백본 네트워크(예: ResNet 또는 HRNet)에서 출력된 중간 표현으로부터 계산되며, 훈련 중 지상 진실 분할(ground-truth segmentation)을 사용하여 교차 엔트로피 손실(cross-entropy loss)로 학습됩니다.

2. **객체 영역 표현 계산 (Object Region Representations):**

   - 각 객체 영역의 표현 $f_{k}$는 해당 객체 영역에 속하는 픽셀들의 표현 $x_{i}$를, 픽셀이 해당 영역에 속하는 정규화된 정도 $\tilde{m}_{ki}$를 가중치로 사용하여 집계함으로써 계산됩니다:
     $$f_{k} = \sum_{i \in I} \tilde{m}_{ki} x_{i}$$
   - 여기서 $\tilde{m}_{ki}$는 공간 소프트맥스(spatial softmax)를 통해 정규화됩니다.

3. **객체 컨텍스트 표현 및 증강 표현 계산 (Object Contextual Representations and Augmented Representations):**
   - 각 픽셀 $p_{i}$와 각 객체 영역 $k$ 간의 관계 $w_{ik}$를 계산합니다:
     $$w_{ik} = \frac{e^{\kappa(x_{i}, f_{k})}}{\sum_{K}_{j=1} e^{\kappa(x_{i}, f_{j})}}$$
     여기서 $\kappa(x,f) = \phi(x)^{T}\psi(f)$이며, $\phi(\cdot)$와 $\psi(\cdot)$는 1x1 conv → BN → ReLU로 구현된 변환 함수입니다.
   - 픽셀 $p_{i}$에 대한 객체 컨텍스트 표현 $y_{i}$는 모든 객체 영역 표현의 가중치 합으로 계산됩니다:
     $$y_{i} = \rho\left(\sum_{K}_{k=1} w_{ik} \delta(f_{k})\right)$$
     여기서 $\delta(\cdot)$와 $\rho(\cdot)$도 1x1 conv → BN → ReLU로 구현된 변환 함수입니다.
   - 최종 픽셀 표현 $z_{i}$는 원래 픽셀 표현 $x_{i}$와 객체 컨텍스트 표현 $y_{i}$를 융합하여 업데이트됩니다:
     $$z_{i} = g([x_{i}^{T} \ y_{i}^{T}]^{T})$$
     여기서 $g(\cdot)$는 1x1 conv → BN → ReLU로 구현된 변환 함수입니다.

**Segmentation Transformer로 재해석:**
OCR 파이프라인은 Transformer의 어텐션 메커니즘으로 재해석될 수 있습니다:

- **디코더 크로스-어텐션:** 소프트 객체 영역 추출 및 객체 영역 표현 계산을 담당합니다. 이미지 특징(key, value)과 $K$개의 카테고리 쿼리(query)를 사용하여 각 카테고리에 대한 어텐션 가중치($\tilde{m}_{ki}$)를 생성하고, 이를 통해 객체 영역 표현($f_k$)을 계산합니다.
- **인코더 크로스-어텐션:** 객체 영역 표현을 집계하는 역할을 합니다. 각 위치의 이미지 특징(query)과 디코더 출력(key, value, 즉 객체 영역 표현)을 사용하여 픽셀-객체 관계($w_{ik}$)를 계산하고, 이를 기반으로 객체 컨텍스트 표현($y_i$)을 생성합니다.

## 📊 Results

- **성능:**

  - **Cityscapes:** ResNet-101 기반으로 81.8% (w/o coarse), 82.4% (w/ coarse). HRNet-W48 기반으로 Mapillary 사전 학습 후 SegFix와 결합하여 84.5% 달성 (ECCV 2020 제출 마감 기준 1위).
  - **ADE20K:** ResNet-101 기반으로 45.28%, HRNet-W48 기반으로 45.66% 달성.
  - **LIP:** ResNet-101 기반으로 55.60%, HRNet-W48 기반으로 56.65% 달성.
  - **PASCAL-Context:** ResNet-101 기반으로 54.8%, HRNet-W48 기반으로 56.2% 달성.
  - **COCO-Stuff:** ResNet-101 기반으로 39.5%, HRNet-W48 기반으로 40.5% 달성.
  - **COCO Panoptic Segmentation:** Panoptic-FPN에 OCR을 적용하여 ResNet-101 백본으로 PQ(Panoptic Quality) 43.0%에서 44.2%로 향상 (mIoU 및 PQ$_{St}$에서 큰 개선).

- **효율성:** 기존의 다중 스케일 컨텍스트(PPM, ASPP) 및 관계형 컨텍스트(DANet, CC-Attention, Self-Attention) 방법들과 비교하여 파라미터 수, GPU 메모리, FLOPs, 추론 시간 면에서 우수성을 보였습니다. 특히, OCR은 대부분의 비교 대상보다 적은 자원 소모로 더 나은 성능을 달성하여 성능과 효율성 사이의 균형이 뛰어남을 입증했습니다. 예를 들어, FLOPs는 PPM의 1/2, DANet의 3/10 수준입니다.

## 🧠 Insights & Discussion

- **핵심 통찰:** 픽셀의 레이블이 본질적으로 픽셀이 속한 객체의 레이블이라는 통찰력에 기반하여, 픽셀 표현을 해당 객체 영역의 표현으로 강화하는 것이 중요함을 입증했습니다. 이는 객체 수준의 컨텍스트를 명시적으로 모델링하는 것이 분할 성능 향상에 매우 효과적임을 시사합니다.
- **영역 감독의 중요성:** 소프트 객체 영역 형성 과정에 대한 지상 진실 감독(supervision)이 분할 성능에 결정적인 역할을 한다는 것을 경험적으로 보여주었습니다. 이는 의미론적 정보가 컨텍스트 집약 과정에 통합될 때 더 정확한 관계 학습이 가능함을 의미합니다.
- **픽셀-영역 관계 계산의 우월성:** 픽셀 표현뿐만 아니라 영역 표현을 함께 사용하여 픽셀-영역 관계를 계산하는 방식이, 단순히 픽셀 표현만 사용하거나 고정된 중간 분할 맵을 사용하는 방식보다 우수함을 확인했습니다. 이는 특정 이미지의 객체 특성을 더 정확하게 포착할 수 있기 때문입니다.
- **Transformer와의 연관성:** 제안된 OCR 방법이 Transformer의 인코더-디코더 프레임워크와 밀접하게 관련되어 있음을 보여줌으로써, 추후 Transformer 기반 모델에 OCR 아이디어를 통합할 수 있는 가능성을 제시합니다. 이는 컨텍스트 모델링에 대한 새로운 관점을 제공합니다.
- **제한 및 향후 연구:** 논문 자체에서 명시적인 한계를 언급하지는 않지만, 객체 영역 형성의 정확도가 전체 시스템 성능에 큰 영향을 미칠 수 있습니다. 또한, Transformer 기반으로의 재해석은 더 큰 모델이나 데이터셋에 대한 스케일링 가능성을 열어줄 수 있습니다.

## 📌 TL;DR

의미론적 분할에서 픽셀-객체 컨텍스트 활용을 위해, 각 픽셀을 해당 객체 클래스 표현으로 특징화하는 **객체 컨텍스트 표현(OCR)**을 제안한다. OCR은 소프트 객체 영역 학습, 객체 영역 표현 계산, 그리고 픽셀-객체 관계 기반의 객체 컨텍스트 표현 증강의 3단계로 이루어진다. 이 방식은 **Transformer 인코더-디코더** 구조로 재해석될 수 있으며, 다양한 벤치마크에서 기존 방법들을 능가하는 성능과 높은 효율성을 동시에 달성한다. 핵심은 픽셀이 속한 객체 정보를 명시적으로 모델링하여 분할 정확도를 높이는 데 있다.
