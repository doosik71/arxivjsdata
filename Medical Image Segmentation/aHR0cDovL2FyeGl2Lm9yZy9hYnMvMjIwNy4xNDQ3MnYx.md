# Beyond CNNs: Exploiting Further Inherent Symmetries in Medical Image Segmentation

Shuchao Pang, Anan Du, Mehmet A. Orgun, Yan Wang, Quan Z. Sheng, Shoujin Wang, Xiaoshui Huang, and Zhenmei Yu

## 🧩 Problem to Solve

본 논문은 의료 영상 분석의 핵심 단계인 종양 및 병변 분할(segmentation)에서 기존 Convolutional Neural Networks (CNNs)가 가진 구조적 한계를 해결하고자 한다. 인간의 시각 시스템은 2D 이미지 내의 대칭성(symmetries)을 효과적으로 감지할 수 있는 반면, 일반적인 CNN은 오직 평행 이동 불변성(translation invariance)만을 활용할 수 있다. 이로 인해 의료 영상에 내재된 회전(rotation) 및 반전(reflection)과 같은 대칭성을 충분히 활용하지 못한다.

이러한 한계는 실제 임상 환경에서 심각한 문제로 이어진다. 예를 들어, 동일한 환자의 CT 슬라이스를 회전시켜 입력했을 때, 일반적인 CNN 기반 모델은 일관되지 않은 예측 결과를 생성하는 경향이 있다. 또한, 의료 영상 데이터는 종양과 주변 조직 간의 낮은 대비, 형태 및 크기의 예측 불가능성, 제한된 데이터 양 등의 어려움이 있어, 단순한 모델 구조로는 정밀한 경계 묘사(delineation)가 어렵다는 문제가 존재한다. 따라서 본 논문의 목표는 의료 영상의 내재적 대칭성을 인코딩하여 더욱 정밀한 표현(representation)을 학습할 수 있는 새로운 Group Equivariant Segmentation 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 일반적인 CNN의 평행 이동 대칭성을 확장하여, 평행 이동, 회전, 반전을 모두 포함하는 **Symmetry Group(대칭 그룹)**을 정의하고 이를 네트워크 전체에 적용하는 것이다.

주요 기여 사항은 다음과 같다.
1. **Group Equivariant Segmentation CNNs 제안**: 일반적인 CNN을 넘어 의료 영상의 내재적 대칭성을 각 레이어에서 효과적으로 캡처할 수 있는 프레임워크를 설계하였다.
2. **전역적 등가성(Global Equivariance) 보장**: 레이어별 대칭 제약 조건(layer-wise symmetry constraints)을 설계하여 네트워크 전체가 전역적으로 Equivariant 하도록 보장하였다. 이를 통해 가중치 공유(weight sharing)를 극대화하고 모델의 표현 능력을 향상시켰다.
3. **GER-UNet 아키텍처 구현**: 제안한 그룹 레이어들을 ResNet-UNet 구조에 통합한 GER-UNet을 통해, 종양의 위치 파악을 넘어 어려운 과제인 '경계 묘사' 성능을 획기적으로 개선하였다.

## 📎 Related Works

논문에서는 기존의 대칭성 해결 방식을 세 가지 범주로 나누어 설명하며 각각의 한계를 지적한다.

1. **Data Augmentation**: 가장 널리 쓰이는 방법으로, 회전 및 반전된 이미지를 추가하여 근사적인 불변성(approximate invariance)을 학습시킨다. 그러나 이는 필터의 중복성을 높여 모델 크기를 키우고 과적합(overfitting) 위험을 증가시키며, 테스트 데이터에 대한 불변성을 수학적으로 보장하지 못한다.
2. **Rotation Equivariant Networks**: 특징 맵(feature map)을 직접 회전시켜 다수의 회전된 특징 맵을 유지하는 방식이다. 이는 구현이 쉽지만, 각 레이어에서 특징 맵을 반복적으로 회전시키므로 메모리 요구량이 급격히 증가한다.
3. **Filter-based Equivariance**: 컨볼루션 커널(kernel) 자체를 회전시키는 방식이다. 로컬 대칭성은 달성할 수 있으나, 차원 폭발 문제와 orientation pooling 과정에서 발생하는 노이즈로 인해 네트워크를 깊게 쌓기 어렵고 전역적 등가성을 유지하기 힘들다는 단점이 있다.

본 논문은 이러한 한계를 극복하기 위해 Cohen과 Welling이 제안한 Group Equivariant CNN의 이론을 이미지 분류에서 시맨틱 분할(semantic segmentation) 영역으로 확장하여 적용하였다.

## 🛠️ Methodology

본 논문은 평행 이동, $\pi/2$ 배수 회전, 그리고 반전을 포함하는 대칭 그룹 $\mathcal{G}$를 정의하고, 이를 기반으로 한 핵심 모듈들을 제안한다.

### 1. 전체 파이프라인 및 핵심 모듈
전체 시스템은 입력부터 출력까지 모든 과정이 동일한 대칭 그룹 $\mathcal{G}$를 따르도록 설계되었다.

*   **Group Input Layer ($\mathbb{Z}^2 \to \mathcal{G}$)**: 
    입력 이미지(2D)를 8개의 회전 및 반전된 버전의 동일한 커널로 컨볼루션 한다. 결과적으로 출력은 $\mathcal{G}$ 그룹 상의 함수가 되며, 이는 입력의 변형이 출력의 동일한 변형으로 이어진다는 Equivariance 성질을 가진다.
    $$[f * \psi_i^{(1)}](g) = \sum_{y \in \mathbb{Z}^2} \sum_{k} f_k(y) \psi_{i,k}^{(1)}(g^{-1}y)$$
*   **Group Hidden Layer ($\mathcal{G} \to \mathcal{G}$)**: 
    이전 레이어의 출력인 그룹 특징 맵을 다시 그룹 컨볼루션 한다. 각 방향(orientation)에 맞는 커널들이 설계되어 있으며, 모든 방향에 대해 대칭 그룹 연산을 수행하여 전역적 등가성을 유지한다.
    $$[f * \psi_i^{(t)}](g) = \sum_{h \in \mathcal{G}} \sum_{k} f_k(h) \psi_{i,k}^{(t)}(g^{-1}h)$$
*   **Group Up-sample Layer**: 
    전통적인 업샘플링(nearest, bilinear)을 8개 모든 방향의 특징 맵에 개별적으로 적용하거나, 전체를 통합하여 보간한 후 다시 분리한다. 이를 통해 디코더 단계에서도 등가성을 유지한다.
*   **Group Skip Connections**: 
    인코더와 디코더의 특징 맵을 결합할 때, 동일한 방향(orientation)끼리 더하거나 연결(concatenate)한다. 이는 각 대칭 속성에서 세부 특징을 복원하여 더 정확한 예측을 가능하게 한다.
*   **Group Output Layer ($\mathcal{G} \to \mathbb{Z}^2$)**: 
    최종 단계에서는 8개의 방향 채널을 하나의 2D 예측 맵으로 통합해야 한다. 본 논문은 전역 평균 풀링(globally average pooling)을 사용하여 모든 방향의 출력을 합산함으로써 최종적인 등가성을 완성한다.
    $$f(x) = \frac{1}{\|\mathcal{G}\|} \sum_{g \in \mathcal{G}} f(g)$$

### 2. GER-UNet 아키텍처
제안된 GER-UNet은 ResNet 블록을 기반으로 한 UNet 구조에 위에서 정의한 그룹 모듈들을 통합한 형태이다. 모든 컨볼루션, 배치 정규화(Batch Normalization), 활성화 함수(ReLU)가 그룹 등가성을 유지하도록 설계되었으며, 인코더-디코더 구조를 통해 다해상도 특징을 효과적으로 추출한다.

## 📊 Results

### 1. 실험 설정
*   **데이터셋**: 공공 간 종양 분할 데이터셋(LITS)을 사용하였으며, 131개의 3D CT 스캔 중 종양 정보가 포함된 7,190개 슬라이스를 대상으로 하였다.
*   **평가 지표**: Dice, Hausdorff distance, Jaccard, Precision, Specificity, F1 score를 사용하여 다각도로 평가하였다.
*   **비교 대상**: U-Net 및 그 변형(Attention UNet, Nested UNet), 컨텍스트 기반 모델(R2U-Net, CE-Net), 어텐션 기반 모델(SENet, DANet, CS-Net) 등 9가지 SOTA 모델과 비교하였다.

### 2. 주요 결과
*   **간 종양 분할 성능**: GER-UNet은 모든 지표에서 비교 모델들을 압도하였다. 특히 Dice 계수 86.63%, Jaccard 80.31%, Precision 87.23%를 기록하며 가장 우수한 성능을 보였다. baseline인 Regular R-UNet 대비 Dice 기준 약 4.03%p 향상되었다.
*   **등가성 검증**: 시각적 분석 결과, 일반 CNN은 입력 이미지를 회전시켰을 때 예측 결과가 크게 변하는 반면, GER-UNet은 입력의 회전에 따라 출력 또한 동일하게 회전하는 강건한 등가성을 보였다. 또한, 종양의 경계(boundary) 묘사가 훨씬 정밀해졌음을 확인하였다.
*   **학습 효율성**: 일반 CNN 기반 모델들이 약 300 epoch 후에 수렴하는 것과 달리, GER-UNet은 약 80 epoch 만에 빠르게 수렴하는 모습을 보였다. 이는 대칭성 인코딩을 통한 가중치 공유가 학습 효율을 크게 높였음을 시사한다.

### 3. 일반화 능력 테스트
본 프레임워크의 범용성을 확인하기 위해 추가 실험을 수행하였다.
*   **COVID-19 폐 감염 분할**: 훈련 데이터가 매우 적은 few-shot 상황(20% 학습, 80% 테스트)에서도 scene segmentation 모델(PSPNet, DeepLabv3+ 등)보다 안정적인 성능을 보였다.
*   **망막 혈관 검출 (DRIVE 데이터셋)**: 매우 얇은 혈관을 검출해야 하는 과제에서 Sensitivity(민감도) 85.59%를 기록하며 특화된 기존 방법들보다 뛰어난 성능을 보였다.
*   **타 모달리티 확장**: MRI(전립선 분할) 및 초음파(갑상샘 분할) 데이터셋에서도 하이퍼파라미터 수정 없이 우수한 일반화 성능을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 의료 영상 분할에서 **"종양의 위치를 찾는 것은 쉽지만, 경계를 정밀하게 묘사하는 것은 어렵다"**는 병목 지점을 정확히 짚어냈다. 이를 해결하기 위해 단순한 데이터 증강이 아닌, 네트워크 구조 자체에 수학적 대칭성을 내재화함으로써 모델의 파라미터 효율성을 높이고 예측의 일관성을 확보하였다.

**강점**:
- 수학적 근거를 바탕으로 전역적 등가성을 보장하여 데이터 효율성을 극대화하였다.
- 특정 장기나 모달리티에 국한되지 않고 다양한 의료 영상(CT, MRI, US, Fundus)에서 성능 향상을 입증하였다.
- 필터 수의 중복성을 줄이면서도 표현 능력을 유지하거나 향상시켰다.

**한계 및 논의**:
- 본 연구는 $\pi/2$ 단위의 회전(90도 단위)에 집중하였으나, 실제 의료 영상에서는 더 미세한 각도의 회전이 존재할 수 있다. 논문의 저자들 또한 향후 연구에서 회전 각도를 줄이는 방향을 탐색하겠다고 언급하였다.
- 대부분 2D 슬라이스 기반으로 진행되었으므로, 3D 볼륨 데이터에 대한 완전한 Group Equivariant 모델로의 확장이 필요하다.

## 📌 TL;DR

이 논문은 의료 영상의 회전 및 반전 대칭성을 수학적으로 모델링한 **Group Equivariant Segmentation 프레임워크**와 이를 적용한 **GER-UNet**을 제안한다. 제안된 방법은 기존 CNN이 가지지 못한 전역적 등가성을 보장하여, 데이터 효율성을 높이고 특히 종양의 정밀한 경계 묘사 성능을 획기적으로 개선한다. 간 종양, COVID-19 폐 감염, 망막 혈관, 전립선 및 갑상샘 분할 등 다양한 과제와 모달리티에서 SOTA 성능을 달성함으로써 의료 영상 분석 분야에서 대칭성 활용의 중요성을 입증하였다.