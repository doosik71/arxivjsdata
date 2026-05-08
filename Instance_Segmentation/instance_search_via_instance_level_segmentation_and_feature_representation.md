# Instance Search via Instance Level Segmentation and Feature Representation

Yu Zhan, Wan-Lei Zhao (2019)

## 🧩 Problem to Solve

본 논문은 특정 시각적 인스턴스(물체, 랜드마크, 사람 등)가 포함된 이미지나 비디오를 검색하는 **Instance Search** 문제를 해결하고자 한다. 이는 단순히 시각적으로 유사한 이미지를 찾는 기존의 Content-Based Image Retrieval (CBIR)과 달리, 쿼리로 주어진 특정 객체와 동일한 물리적 개체(same instance)를 찾아내야 하며, 검색된 이미지 내에서 해당 객체의 위치를 정확히 특정해야 한다는 점에서 차별화된다.

기존의 Instance Search 방식들은 SIFT, SURF와 같은 Local Feature에 의존하였으나, 이는 평면 외 회전(out-of-plane rotation)이나 변형에 취약하며, 특히 투명하거나 표면이 매끄러운 물체에서 특징점 추출이 어렵다는 한계가 있다. 또한, 추출된 특징점들이 배경 정보에 오염되어 인스턴스 고유의 표현력을 떨어뜨리는 문제가 발생한다. 최근의 CNN 기반 글로벌 특징 추출 방식 역시 이미지 전체의 활성화 맵을 사용하므로, 하나의 이미지에 여러 인스턴스가 존재할 경우 특징이 섞이게 되어 정밀한 인스턴스 표현이 불가능하다는 문제가 있다. 따라서 본 논문의 목표는 배경의 간섭을 최소화하고 인스턴스 자체의 특성만을 정밀하게 반영하는 **Instance-level feature representation**을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Fully Convolutional Instance-aware Segmentation (FCIS)**를 활용하여 픽셀 수준에서 인스턴스를 정밀하게 분리하고, 이렇게 분리된 영역 내에서만 특징을 추출함으로써 배경 오염을 원천적으로 차단하는 것이다.

주요 기여 사항은 다음과 같다.

1. **Instance-level Feature Representation**: FCIS를 통해 인스턴스를 세그멘테이션하고, 해당 영역에 대해 ROI Pooling을 수행하여 다양한 크기와 레이아웃의 인스턴스를 동일한 길이의 벡터로 표현하는 파이프라인을 제안하였다.
2. **Network Enhancement**: 기본 FCIS 구조를 개선하기 위해 백본 네트워크를 **ResNeXt-101**으로 교체하여 표현력을 높였으며, 마지막 세 개의 bottleneck 블록에 **Deformable Convolution**을 적용하여 객체의 기하학적 변형에 강건하게 대응하도록 설계하였다.
3. **Instance-160 Dataset**: Instance Search를 위한 공개 벤치마크의 부족을 해결하기 위해, 비디오 객체 추적(Visual Object Tracking) 데이터셋인 OTB2015와 ALOV++를 기반으로 한 새로운 평가 데이터셋인 **Instance-160**을 구축하였다.

## 📎 Related Works

기존의 접근 방식은 크게 두 가지로 나뉜다. 첫째는 BoVW (Bag-of-Visual Words), VLAD, Fisher Vector (FV)와 같이 SIFT 등의 Local Feature를 집계하는 방식이다. 이들은 구별력은 높지만 변형에 취약하고 배경 노이즈에 민감하다는 한계가 있다. 둘째는 최근의 CNN 기반 이미지 검색 방식으로, 이미지 전체 혹은 특정 영역의 feature map을 사용하는 방식이다. 하지만 이들은 여전히 Global feature의 성격을 띠고 있어, 여러 객체가 섞여 있는 환경에서 특정 인스턴스만을 정밀하게 구분해내는 능력이 부족하다.

본 논문은 이러한 한계를 극복하기 위해 '세그멘테이션 $\rightarrow$ 특징 추출'의 순차적 구조를 채택하여, 특징 추출 단계 이전에 이미 인스턴스 영역을 픽셀 수준에서 확정함으로써 기존 방식들이 겪었던 배경 오염 문제를 해결하였다.

## 🛠️ Methodology

### 1. Instance-level Feature Extraction Pipeline

전체 시스템은 FCIS 네트워크를 기반으로 하며, 다음과 같은 절차로 인스턴스 특징을 추출한다.

- **인스턴스 분리**: FCIS의 Region Proposal Network (RPN)와 position-sensitive score map을 통해 이미지 내 각 인스턴스의 바운딩 박스(Bounding Box)와 픽셀 수준의 마스크(Mask)를 생성한다.
- **ROI Pooling**: 생성된 바운딩 박스를 기반으로 CNN의 컨볼루션 레이어 feature map에서 ROI Pooling을 수행한다. 이때 각 채널에서 최대 활성화 값(maximum activation)을 추출하여 하나의 차원으로 사용한다.
- **특징 벡터 생성**: 결과적으로 각 인스턴스는 feature map의 채널 수와 동일한 길이의 고정된 벡터로 표현된다. 이는 인스턴스의 크기에 상관없이 일관된 길이의 특징량을 제공한다.
- **하이브리드 특징(Hybrid Features)**: 단일 레이어보다 여러 레이어의 특징을 결합하는 것이 효과적임을 확인하였으며, 특히 `conv3`와 `conv4` 레이어의 특징을 연결(concatenation)한 후 $L_2$ 정규화를 거쳐 최종 표현량으로 사용한다.

### 2. Performance Enhancement (FCIS+XD)

네트워크의 성능을 극대화하기 위해 두 가지 구조적 변경을 가하였다.

- **ResNeXt-101 도입**: 기존 ResNet-101을 ResNeXt-101로 교체하였다. ResNeXt는 cardinality(동일한 토폴로지의 변환 집합의 크기)를 높임으로써 연산량(FLOPs)의 증가 없이 더 높은 정확도를 달성한다. 본 논문에서는 cardinality를 32로 설정하였다.
- **Deformable Convolution 적용**: 일반적인 컨볼루션은 고정된 그리드에서 샘플링하므로 객체의 변형에 취약하다. 이를 해결하기 위해 ResNeXt-101의 마지막 3개 bottleneck 블록에 Deformable Convolution을 적용하였다. 이는 입력 feature map에 대해 오프셋(offset)을 학습하여 샘플링 위치를 유연하게 조정함으로써, 다양한 형태의 객체에 적응적인 수용 영역(receptive field)을 갖게 한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 자체 구축한 **Instance-160** (쿼리 이미지 및 11,885장의 참조 이미지)와 확장성 테스트를 위한 Flickr 이미지 100만 장을 사용하였다.
- **비교 대상**: BoVW, BoVW+HE, R-MAC, Deepvision (DV-Vgg, DV-Res), CroW.
- **평가 지표**: mAP @ top-k ($k \in \{10, 20, 50, 100\}$) 및 All-rank mAP.

### 2. 주요 결과

- **구조 개선 효과**: FCIS 기본 모델보다 ResNeXt와 Deformable Conv가 모두 적용된 **FCIS+XD** 모델이 인스턴스 세그멘테이션(PASCAL VOC, COCO)과 Instance Search 모두에서 가장 높은 성능을 보였다.
- **Instance Search 성능**: Table 4에 따르면 FCIS+XD는 모든 랭킹 지표에서 기존 딥러닝 기반 방식(R-MAC, CroW)과 전통적 방식(BoVW)을 압도하였다. 특히 DV-Vgg와 유사한 성능을 보였으나, 배경 변형이 심한 데이터셋(Table 5)에서는 FCIS+XD가 훨씬 강력한 성능을 유지하였다.
- **확장성(Scalability)**: 100만 장의 방해 이미지(distractors)를 추가한 실험에서 FCIS+XD는 다른 모든 방법론보다 높은 mAP@50를 기록하며, 대규모 데이터셋에서도 인스턴스를 구별해내는 능력이 뛰어남을 입증하였다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 강점은 **'정밀한 세그멘테이션이 정밀한 특징 표현으로 이어진다'**는 직관을 증명한 것이다. 기존의 Deepvision과 같은 방식은 이미지 수준의 특징으로 1차 필터링을 수행하므로, 쿼리 이미지와 참조 이미지의 배경이 유사할 경우 실제로는 다른 인스턴스임에도 불구하고 높은 순위로 검색되는 '배경 의존성' 문제가 발생한다. 반면, FCIS+XD는 인스턴스 영역만을 분리하여 특징을 추출하므로 배경의 영향에서 자유롭다.

다만, 본 모델의 한계점은 FCIS가 Microsoft COCO의 80개 카테고리에 종속되어 있다는 점이다. 즉, 학습된 80개 클래스에 속하지 않는 일반적인 물체에 대해서는 세그멘테이션 성능이 떨어지므로 Instance Search 성능 또한 제한될 수 있다. 향후 연구에서는 클래스 제약이 없는 더 일반적인(generic) 인스턴스 세그멘테이션 모델을 적용하는 것이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 FCIS를 통해 인스턴스를 픽셀 수준으로 정밀하게 분리하고, 해당 영역에서 ROI Pooling을 통해 특징을 추출하는 **Instance-level feature representation** 방식을 제안하였다. 여기에 **ResNeXt-101** 백본과 **Deformable Convolution**을 추가하여 변형에 강건한 표현력을 확보하였으며, 새롭게 구축한 **Instance-160** 데이터셋과 100만 장의 확장성 테스트를 통해 기존 방식 대비 우수한 구별력과 확장성을 입증하였다. 이 연구는 객체 중심의 정밀한 특징 추출이 배경 노이즈를 제거하고 검색 정확도를 높이는 핵심임을 보여주며, 향후 범용 인스턴스 검색 시스템 구축에 중요한 기초를 제공한다.
