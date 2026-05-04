# Hybrid Instance-aware Temporal Fusion for Online Video Instance Segmentation

Xiang Li, Jinglu Wang, Xiao Li, Yan Lu (2022)

## 🧩 Problem to Solve

본 논문은 온라인 비디오 인스턴스 분할(Online Video Instance Segmentation, VIS)에서 프레임 간(Frame-to-Frame, F2F) 통신을 효과적으로 모델링하는 문제를 해결하고자 한다. VIS는 각 프레임에서 객체를 분류(Classification), 분할(Segmentation)함과 동시에 동일한 객체에 동일한 ID를 부여하는 추적(Tracking)을 동시에 수행해야 하는 고난도 작업이다.

기존의 온라인 VIS 방법론들은 주로 픽셀 수준(Pixel-level)에서만 통신을 모델링하여 고수준의 인스턴스 정보를 반영하는 데 한계가 있거나, 검출된 박스(Bounding Box)를 기반으로 RoI(Region of Interest) 특징을 추출하여 융합하는 방식을 사용하였다. 그러나 박스 기반 방식은 객체 검출기의 정확도에 지나치게 의존하며, 크롭된 특징들이 전역 문맥(Global Context)으로부터 고립되어 정보가 불완전하고 편향될 수 있다는 치명적인 단점이 있다. 따라서 본 논문의 목표는 검출 및 크롭 과정 없이, 전역 문맥을 유지하면서 인스턴스와 픽셀 수준의 정보를 동시에 활용하는 새로운 하이브리드 시간적 융합(Hybrid Temporal Fusion) 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 전역 문맥 내의 잠재 코드(Latent Code)인 **Instance Code**와 CNN 특징 맵(Feature Map)을 결합하여 인스턴스와 픽셀 수준의 특징을 동시에 표현하는 하이브리드 표현법을 도입하는 것이다.

주요 기여 사항은 다음과 같다.
1. **Box-free 및 Matching-free 구조**: 복잡한 객체 검출-크롭 과정이나 프레임 간의 복잡한 인스턴스 매칭 연산 없이, 인스턴스 코드의 슬롯 인덱스(Slot Index)를 통해 직접적으로 ID를 유지하는 프레임워크를 제안한다.
2. **Hybrid Instance-aware Temporal Fusion**: 인스턴스 코드와 픽셀 특징 맵 사이의 교차 주의 집중(Cross-attention) 메커니즘을 통해 인스턴스 레벨과 픽셀 레벨의 정보를 동시에 융합하는 방법을 제안한다.
3. **Temporal Consistency 강화**: 학습 단계에서 인스턴스 코드의 슬롯 일관성(Slot Consistency)을 강제하는 제약 조건을 통해 프레임 간의 일관성을 높였다.

## 📎 Related Works

비디오 인스턴스 분할 연구는 크게 오프라인(Offline)과 온라인(Online) 방식으로 나뉜다. 
- **오프라인 방식**: 전체 비디오 클립을 입력으로 받아 시공간적 상관관계를 모델링하며 높은 정확도를 보이지만, 실시간 스트리밍 서비스에 적용하기 어렵다. 대표적으로 MaskProp과 VisTR 등이 있으며, 특히 VisTR은 트랜스포머 디코더를 통해 ID를 매칭한다.
- **온라인 방식**: 프레임 단위로 입력을 처리하여 실용적이지만, 프레임 간 통신 모델링의 불완전함으로 인해 오프라인 방식보다 성능이 낮다. MaskTrack R-CNN, SipMask, SG-Net 등이 있으며, 이들은 주로 추적 헤드(Tracking Head)를 추가하여 ID를 연관시킨다.

본 논문은 기존의 온라인 방식들이 픽셀 수준의 통신에만 치중하거나 검출 박스에 의존했다는 한계를 지적하며, MaX-DeepLab의 박스 없는(Box-free) 이미지 분할 구조를 VIS로 확장하여 하이브리드 수준의 통신을 구현함으로써 차별점을 둔다.

## 🛠️ Methodology

### 1. 하이브리드 표현 (Hybrid Representation)
각 프레임의 특징을 표현하기 위해 두 가지 요소를 사용한다.
- **Instance Code ($e$)**: $L \times D$ 크기의 벡터로, $L$은 프레임 내 최대 인스턴스 수, $D$는 특징 차원이다. 각 슬롯은 인스턴스의 클래스와 마스크 정보를 순서 인지(Order-aware) 방식으로 인코딩하며, 이 슬롯 인덱스가 곧 인스턴스의 ID가 된다.
- **CNN Feature Map ($f$)**: 공간적 세부 정보를 담고 있는 픽셀 수준의 특징 맵이다.

### 2. 인스턴스 인지 프레임 통신 (Instance-aware Frame Communication)
트랜스포머 기반의 주의 집중 메커니즘을 통해 프레임 내(Intra-frame) 및 프레임 간(Inter-frame) 정보를 융합한다.

#### Intra-frame Attention
현재 프레임 내에서 인스턴스 코드와 특징 맵 사이의 관계를 구축한다. 세 가지 타입의 어텐션을 수행한다.
- **Code-to-Code (c2c) & Code-to-Pixel (c2p)**: 인스턴스 코드가 쿼리($Q$)가 되어 다른 코드와 픽셀 맵에서 인스턴스 관련 특징을 추출한다.
$$e^{intra} = \sum_{n=1}^{HW+L} \text{softmax}(Q^e_t \cdot K^{e \oplus f}_t) V^{e \oplus f}_t$$
- **Pixel-to-Code (p2c)**: 픽셀 특징 맵이 쿼리가 되어 인스턴스 코드로부터 정보를 받아 픽셀 맵을 조정한다.
$$f^{intra} = \sum_{n=1}^{L} \text{softmax}(Q^f_t \cdot K^e_t) V^e_t$$

#### Inter-frame Attention
타겟 프레임 $I_t$와 참조 프레임 세트 $R=\{I_{t-\delta}\}$ 간의 시간적 상관관계를 모델링한다. 참조 프레임들의 코드와 특징 맵을 연결(Concatenate)하고 위치 인코딩(Positional Encoding)을 추가하여 사용한다.
- **Inter-frame c2c & c2p**: 타겟 프레임의 코드가 참조 프레임의 코드와 픽셀 맵에서 정보를 추출하여 시간적 문맥을 융합한다.
$$e^{inter} = \sum_{n=1}^{HW+L} \text{softmax}(Q^{e}_{tgt} \cdot K^{e \oplus f}_{ref}) V^{e \oplus f}_{ref}$$
- **Inter-frame p2c**: 타겟 프레임의 픽셀 맵이 참조 프레임의 인스턴스 코드로부터 정보를 추출한다.
$$f^{inter} = \sum_{n=1}^{L} \text{softmax}(Q^{f}_{tgt} \cdot K^{e}_{ref}) V^{e}_{ref}$$

### 3. 네트워크 구조 및 디코더 (Network Design & Decoder)
- **Encoder**: ResNet-50 백본의 마지막 단계에 Intra-frame Attention을 삽입하고, 이후 Inter-frame과 Intra-frame Attention을 $M$번 반복하여 특징을 추출한다.
- **Decoder**: Deeplab-V3+ 구조를 사용하여 저수준 특징을 융합한다.
- **Mask Prediction**: 인스턴스 코드를 통해 동적 필터(Dynamic Filter) $\theta^t$를 생성하고, 이를 업샘플링된 특징 맵 $f^{out}$에 적용하는 동적 컨볼루션(Dynamic Convolution)을 통해 마스크를 생성한다.
$$M^t = \text{softmax}(\theta^t f^{T}_{out})$$

### 4. 손실 함수 및 학습 절차 (Loss Function)
인스턴스 ID의 일관성을 유지하기 위해 현재 프레임($t$)과 이전 프레임($t-1$)의 예측값을 동시에 고려하여 최적의 순열(Permutation) $\sigma$를 찾아 Ground-truth와 매칭한다. 유사도는 마스크의 Dice 계수와 클래스 예측 확률의 곱으로 계산한다.
전체 손실 함수는 다음과 같다.
$$L = \lambda_{inst} L_{pos} + (1 - \lambda_{inst}) L_{neg} + \lambda_{aux} L_{aux}$$
여기서 $L_{pos}$는 매칭된 인스턴스의 클래스 및 마스크 손실이며, $L_{neg}$는 매칭되지 않은 슬롯이 빈 클래스($\emptyset$)를 예측하도록 하는 손실이다.

## 📊 Results

### 실험 설정
- **데이터셋**: Youtube-VIS-2019 및 Youtube-VIS-2021
- **평가 지표**: mAP (mean Average Precision)
- **구현 세부사항**: ResNet-50 백본 사용, COCO 데이터셋으로 사전 학습 후 파인튜닝, 참조 프레임 수는 3개로 설정.

### 주요 결과
- **정량적 결과**: Youtube-VIS-2019에서 **41.3 mAP**를 기록하여 기존의 모든 온라인 VIS 방법론을 크게 상회하였다. 특히, 동일한 ResNet-50 백본을 사용한 오프라인 방법론들보다도 우수한 성능을 보였으며, 더 강력한 ResNet-101 백본을 사용한 CrossVIS보다 높은 성능을 달성하였다. Youtube-VIS-2021에서도 **35.8 mAP**로 SOTA 성능을 기록하였다.
- **정성적 결과**: 시각화 결과, VisTR과 같은 기존 모델이 객체가 겹치거나 작은 경우 인스턴스를 놓치는 반면, 제안 모델은 매우 도전적인 시나리오에서도 정확하고 일관된 추적 및 분할 성능을 보였다.

### 절제 연구 (Ablation Study)
- **Inter-frame Attention**: 모든 시간적 어텐션을 제거했을 때 성능이 36.9 mAP까지 급락하여, 하이브리드 융합의 중요성이 입증되었다.
- **참조 프레임 수**: 참조 프레임 수를 1개에서 3개로 늘릴 때 mAP가 39.7에서 41.3으로 증가하여, 더 많은 시간적 문맥이 유리함을 확인하였다.
- **슬롯 수**: 인스턴스 코드의 슬롯 수를 10개에서 25개까지 늘려도 성능이 안정적으로 유지되어, 중복 슬롯에 대해 강건함을 보였다.

## 🧠 Insights & Discussion

본 논문은 VIS 작업에서 '검출-크롭-매칭'으로 이어지는 기존의 파이프라인이 가진 의존성과 비효율성을 지적하고, 이를 트랜스포머의 쿼리(Query) 기반 메커니즘으로 대체함으로써 문제를 단순화하고 성능을 높였다. 

특히, **Instance Code**라는 전역 잠재 변수를 도입하여 이를 ID의 대리자로 사용한 점이 매우 영리한 설계이다. 이는 복잡한 매칭 알고리즘 없이도 슬롯 인덱스만으로 ID를 유지할 수 있게 하여 연산 비용을 줄이면서도, 전역 문맥을 유지한 채 픽셀 수준의 세부 정보를 융합할 수 있게 하였다. 

다만, 슬롯 기반의 ID 유지 방식은 매우 급격한 객체의 등장/소멸이 반복되는 환경에서 슬롯의 재할당 문제가 발생할 가능성이 있으며, 이에 대한 구체적인 처리 메커니즘(IoU 0.5 기준의 보완책 외에)에 대한 더 깊은 논의가 필요할 것으로 보인다.

## 📌 TL;DR

본 연구는 온라인 비디오 인스턴스 분할(VIS)을 위해 **박스 검출과 복잡한 ID 매칭 과정이 필요 없는 하이브리드 시간적 융합 프레임워크**를 제안한다. 전역 인스턴스 코드(Instance Code)와 픽셀 특징 맵을 동시에 활용하는 교차 주의 집중(Cross-attention) 메커니즘을 통해 시간적 일관성을 효과적으로 학습하였으며, 그 결과 Youtube-VIS-2019/2021 벤치마크에서 기존 온라인 및 일부 오프라인 모델을 뛰어넘는 SOTA 성능을 달성하였다. 이는 향후 실시간 비디오 분석 시스템에서 효율적이고 정확한 객체 추적 및 분할을 구현하는 데 중요한 기반이 될 것으로 기대된다.