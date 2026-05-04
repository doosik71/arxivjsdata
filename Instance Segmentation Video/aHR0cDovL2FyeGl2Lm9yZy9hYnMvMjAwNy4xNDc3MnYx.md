# SipMask: Spatial Information Preservation for Fast Image and Video Instance Segmentation

Jiale Cao, Rao Muhammad Anwer, Hisham Cholakkal, Fahad Shahbaz Khan, Yanwei Pang, Ling Shao (2020)

## 🧩 Problem to Solve

본 논문은 Single-stage instance segmentation 방법론들이 Two-stage 방법론들에 비해 추론 속도는 빠르고 구조는 단순하지만, 정확도 면에서 여전히 뒤처진다는 점을 해결하고자 한다.

특히, 기존의 대표적인 Single-stage 모델인 YOLACT의 경우, 객체 하나에 대해 단일 세트의 object-aware coefficients를 사용하여 전체 마스크를 예측한다. 이로 인해 바운딩 박스(Bounding-box) 내부의 공간 정보(Spatial information)가 손실되며, 결과적으로 공간적으로 인접해 있거나 겹쳐 있는 객체들을 정밀하게 구분해내지 못하는 한계가 있다.

따라서 본 연구의 목표는 바운딩 박스 내의 공간 정보를 보존함으로써 마스크 예측의 정밀도를 높이고, 특히 인접한 인스턴스 간의 경계를 정확하게 묘사할 수 있는 빠르고 정확한 Single-stage instance segmentation 모델인 SipMask를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 바운딩 박스 내의 마스크 예측을 여러 개의 하위 영역(Sub-regions)으로 분리하여 처리하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Spatial Preservation (SP) Module**: 바운딩 박스를 $k \times k$ 개의 하위 영역으로 나누고, 각 영역에 대해 별도의 spatial coefficients를 생성함으로써 객체 내부의 공간 정보를 보존한다. 이를 통해 인접한 객체 간의 분별력을 높였다.
2. **Mask Alignment Weighting Loss**: 검출된 바운딩 박스의 품질(정확도)이 높을수록 해당 마스크 예측 오류에 더 높은 가중치를 부여하는 손실 함수를 도입하여, 검출과 세그멘테이션 간의 상관관계를 강화하였다.
3. **Feature Alignment Scheme**: Deformable Convolutional layer를 도입하여, 예측된 바운딩 박스의 위치에 맞게 특징 맵(Feature map)을 정렬함으로써 분류 및 계수 생성에 최적화된 특징 표현을 얻도록 하였다.
4. **실시간 비디오 확장성**: 제안된 구조를 기반으로 트래킹 브랜치를 추가하여 실시간 Video Instance Segmentation으로 확장 가능함을 입증하였다.

## 📎 Related Works

기존의 Instance Segmentation은 크게 Top-down 방식의 Two-stage와 Single-stage 방법으로 나뉜다.

- **Two-stage Methods**: Mask R-CNN, PANet 등이 대표적이다. RPN을 통해 제안 영역을 생성하고 RoIAlign과 같은 풀링 연산을 통해 고정 크기의 특징을 추출한다. 정확도는 높지만, 풀링 및 헤드 네트워크의 연산 비용으로 인해 실시간 적용이 어렵다.
- **Single-stage Methods**: YOLACT, TensorMask 등이 있으며, 제안 영역 생성이나 풀링 과정 없이 dense prediction을 수행하여 속도가 빠르다.
  - **YOLACT**는 이미지 전체에 대한 basis masks와 인스턴스별 coefficients의 선형 결합으로 마스크를 생성하여 매우 빠르지만, 앞서 언급한 공간 정보 손실 문제로 인해 정확도가 낮다.
  - **TensorMask**는 높은 정확도를 보이지만 속도가 매우 느려 Two-stage 방법보다도 느린 경우가 발생한다.

SipMask는 YOLACT의 효율성을 유지하면서도, 별도의 특성 풀링(Feature pooling) 없이 spatial coefficients를 세분화함으로써 Two-stage 방법론들이 얻는 공간적 이점을 효율적으로 확보하고자 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

SipMask는 anchor-free 검출기인 FCOS를 기반으로 하며, 기존의 분류 및 회귀 헤드를 **Mask-specialized classification branch**와 **Mask-specialized regression branch**로 대체한다.

### 2. Spatial Preservation (SP) Module

분류 브랜치 내에 위치하며, 다음 두 가지 기능을 수행한다.

- **Spatial Coefficients Generation**: 바운딩 박스를 $k \times k$ (실험적으로 $2 \times 2$가 최적) 영역으로 분할한다. 각 하위 영역 $i$에 대해 별도의 계수 $c_{ij} \in \mathbb{R}^m$를 예측한다.
- **Feature Alignment**: 회귀 브랜치에서 얻은 바운딩 박스 오프셋을 이용해 Deformable Convolutional layer를 적용한다. 입력 특징 $x$에 대해 다음과 같이 정렬된 특징 $y(p_0)$를 얻는다.
$$y(p_0) = \sum_{i \in G} w_r \cdot x(p_0 + p_r + \Delta p_r)$$
여기서 $\Delta p_r$은 학습된 커널 오프셋이며, 이를 통해 예측된 박스 위치에 최적화된 특징을 추출한다.

### 3. Mask-specialized Regression Branch

바운딩 박스 오프셋과 함께 이미지 전체에 적용되는 **Basis Masks**를 생성한다.

- FPN의 $P_3, P_4, P_5$ 레이어 특징을 추출하여 $P_3$ 해상도로 업샘플링 후 결합(Concatenate)한다.
- $3 \times 3$ 컨볼루션을 거쳐 $m$ 채널의 특징 맵을 만들고, 이를 4배 업샘플링하여 최종 $m$개의 basis masks를 생성한다.

### 4. Spatial Mask Prediction (SMP) Module

최종 마스크를 생성하는 과정은 다음과 같다.

1. **선형 결합**: Basis masks $B$와 $i$번째 하위 영역의 계수 $C_i$를 행렬 곱하고 시그모이드 $\sigma$를 적용하여 맵 $M_i$를 생성한다.
$$M_i = \sigma(B \times C_i), \quad \forall i \in [1, 4]$$
2. **Pruning**: 생성된 $M_{ij}$에서 바운딩 박스의 해당 하위 영역(사분면) 외부의 값을 0으로 설정하여 $\hat{M}_{ij}$를 얻는다.
3. **통합 및 이진화**: 모든 하위 영역의 맵을 합산하여 $\hat{M}_j = \sum_{i=1}^4 \hat{M}_{ij}$를 구하고, 임계값(threshold)을 적용해 최종 마스크 $\tilde{M}_j$를 생성한다.

### 5. Loss Function

전체 손실 함수는 $L = L_{\text{reg}} + L_{\text{cls}} + L_{\text{mask}}$이다. 여기서 $L_{\text{mask}}$는 **Mask Alignment Weighting**이 적용된 BCE 손실이다.

- 가중치 $\alpha_j$는 예측 박스와 Ground-truth 간의 겹침 정도(overlap) $o_j$와 분류 점수 $s_j$의 곱으로 계산된다.
$$L_{\text{mask}} = \frac{1}{N} \sum_{j} l_j \times \alpha_j$$
품질이 좋은 바운딩 박스에서 발생한 마스크 오류에 더 큰 가중치를 두어 학습을 유도한다.

### 6. Video Instance Segmentation 확장

실시간 비디오 처리를 위해 트래킹 브랜치를 추가한다. RoIAlign 대신 바운딩 박스 중심점의 특징 벡터를 추출하여 프레임 간 인스턴스를 매칭하는 방식을 사용함으로써 연산 효율성을 극대화하였다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: COCO test-dev 및 YouTube-VIS
- **백본**: ResNet50/101-FPN
- **지표**: Mask AP, 추론 속도(Inference time)

### 2. 정량적 결과 (Image Segmentation)

- **SipMask vs TensorMask**: 동일한 입력 크기와 백본 설정에서 SipMask는 TensorMask보다 **mask AP가 1.0% 높으며, 속도는 4배 더 빠르다**. 특히 대형 객체(Large objects)에서 2.7%의 큰 성능 향상을 보였다.
- **SipMask vs YOLACT**: 실시간 설정(작은 입력 크기)에서 YOLACT 대비 **mask AP가 3.0% 향상**되었으며, 속도는 거의 동일한 수준(YOLACT 30ms vs SipMask 32ms)을 유지하였다.

### 3. 정성적 결과 및 분석

- **공간 분할의 효과**: $1 \times 1$에서 $2 \times 2$로 하위 영역을 늘렸을 때 AP가 크게 상승하였으며, 그 이상의 증가는 성능 향상이 미미하여 $2 \times 2$가 최적의 tradeoff임을 확인하였다.
- **인접 객체 구분**: 시각적 분석 결과, 단일 계수를 사용하는 baseline은 인접한 객체의 픽셀이 마스크에 포함되는 경향이 있으나, SipMask는 영역별 계수를 사용하여 이를 효과적으로 제거하였다.

### 4. 비디오 세그멘테이션 결과

- YouTube-VIS 데이터셋에서 MaskTrack R-CNN 대비 **mask AP가 2.2% 향상**된 32.5%를 기록하였으며, Titan Xp GPU에서 30 fps의 실시간 속도를 달성하였다.

## 🧠 Insights & Discussion

본 논문은 Single-stage 모델의 고질적인 문제인 '공간 정보 손실'을 매우 가벼운 연산(계수의 분할 및 단순 합산)만으로 해결했다는 점에서 큰 강점이 있다.

특히, 복잡한 RoIAlign이나 고해상도 특성 풀링 없이도 $k \times k$ spatial coefficients라는 단순한 설계를 통해 Two-stage 모델에 근접한 경계 묘사 능력을 확보한 점이 인상적이다. 또한, 단순한 BCE loss에 바운딩 박스 품질 가중치($\alpha_j$)를 도입하여 검출과 세그멘테이션의 정렬(alignment)을 꾀한 점은 실용적인 개선책으로 판단된다.

다만, $k$ 값을 높일수록 성능이 소폭 상승하는 경향이 있으나 연산량이 증가하는 trade-off가 존재하며, 본 논문에서는 $2 \times 2$라는 고정된 하이퍼파라미터를 사용하였다. 객체의 크기나 형태에 따라 가변적인 $k$ 값을 적용하는 전략이 있다면 추가적인 성능 향상이 가능할 것으로 보인다.

## 📌 TL;DR

SipMask는 바운딩 박스 내부를 하위 영역으로 나누고 각 영역에 개별적인 spatial coefficients를 적용함으로써, 기존 Single-stage 모델이 놓쳤던 객체 내 공간 정보를 보존하는 방법론이다. 이를 통해 실시간 속도를 유지하면서도 인접 객체 간의 경계를 정밀하게 구분할 수 있게 되었으며, TensorMask보다 4배 빠르고 YOLACT보다 정확한 성능을 보였다. 이 구조는 단순한 브랜치 추가만으로 실시간 비디오 인스턴스 세그멘테이션까지 확장이 가능하여 실제 실시간 시스템 적용 가능성이 매우 높다.
