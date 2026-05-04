# Spatially Dependent U-Nets: Highly Accurate Architectures for Medical Imaging Segmentation

João B. S. Carvalho, João A. Santinha, Ðorđe Miladinović, and Joachim M. Buhmann (2021)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)의 핵심은 관심 영역(Region of Interest, ROI)을 정밀하게 식별하는 것이다. 하지만 기존의 Convolutional Neural Networks (CNN) 기반 아키텍처들은 다음과 같은 두 가지 상충되는 요구사항을 동시에 충족하는 데 어려움을 겪는다.

첫째, 객체의 경계를 정의하는 국소적인 텍스처와 강도 불연속성(intensity discontinuities)을 정확하게 식별해야 한다. 둘째, 영상 전체의 맥락(contextual information)과 의미론적 구조를 파악하여 이미지 내 다른 영역과의 관련성을 평가하는 전역적 정보의 활용이 필요하다.

기존의 U-Net 및 U-Net++와 같은 아키텍처들은 국소적 특징 추출에는 능숙하지만, Convolutional layer의 수용 영역(Receptive Field)이 본질적으로 제한적이라는 한계가 있다. 이로 인해 의료 영상과 같이 해부학적 구조의 공간적 일관성(Spatial Coherence)이 중요한 데이터에서 전역적인 맥락을 놓치게 되며, 결과적으로 잘못된 영역을 ROI로 판단하는 False Positive가 발생하거나 일관성 없는 분할 결과가 도출되는 문제가 발생한다. 본 논문의 목표는 이러한 제한적인 수용 영역 문제를 해결하여 전역적으로 일관된 고정밀 의료 영상 분할 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 최근 생성 모델링 분야에서 제안된 Spatial Dependency Networks (SDN)를 U-Net 구조에 통합하는 것이다. CNN의 Convolutional layer가 커널 크기에 의해 수용 영역이 제한되는 것과 달리, SDN은 이미지 전체를 방향별로 훑는(sweep) 방식을 통해 이론적으로 무제한의 수용 영역(unbounded receptive field)을 가진다.

이를 통해 모델은 픽셀/복셀 공간에서 장거리 공간 의존성(long-range spatial dependencies)을 명시적으로 모델링할 수 있으며, 해부학적 구조가 가지는 공간적 일관성이라는 귀납적 편향(inductive bias)을 네트워크 설계에 반영할 수 있다. 연구팀은 이를 기반으로 두 가지 변형 아키텍처인 **SDU-Net** (Spatially Dependent U-Net)과 **SDNU-Net** (Spatially Dependent Nested U-Net)을 제안하였다.

## 📎 Related Works

기존의 의료 영상 분할에서는 U-Net이 표준적으로 사용되어 왔다. U-Net은 인코더-디코더 구조에 Skip Connection을 도입하여 서로 다른 시맨틱 레벨의 특징을 융합함으로써 성능을 높였다. 이후 등장한 U-Net++는 중첩된 밀집 연결(nested dense connections)을 통해 인코더와 디코더 사이의 시맨틱 갭을 줄이고 멀티스케일 특징을 더 효과적으로 활용하고자 하였다.

그러나 이러한 U-Net 계열의 네트워크들은 여전히 Convolution 기반의 연산에 의존하므로, 비국소적(non-local) 유사성 비교나 전체적인 이미지 일관성을 유지하는 능력이 부족하다는 한계가 있다. 본 논문은 기존의 CNN 방식이 가지는 국소적 수용 영역의 한계를 지적하며, 이를 극복하기 위해 SDN을 도입함으로써 기존 U-Net 계열 아키텍처와 차별화를 꾀하였다.

## 🛠️ Methodology

### 1. Spatial Dependency Layers (SDN Layer)

SDN 레이어는 공간적 일관성과 고차 통계적 상관관계를 모델링하기 위해 다음의 3단계 과정을 거친다.

**가. Project-in Stage**
입력 특징 맵 $X_s$를 학습 가능한 가중치 행렬 $W$와 편향 $b$를 이용해 아핀 변환(affine transformation)을 수행하여 채널 수를 조정한다.
$$X_{s+1}^{i,j} = X_s^{i,j} W + b$$
여기서 $i, j$는 2D 좌표를 의미하며, 이 단계는 입력 표현의 채널 수를 튜닝 가능한 특정 크기로 확장하는 역할을 한다.

**나. Correction Stage**
이 단계가 SDN의 핵심으로, 특징 맵을 네 가지 방향(좌$\rightarrow$우, 우$\rightarrow$좌, 상$\rightarrow$하, 하$\rightarrow$상)으로 스캔한다. 이때 순환 유닛(recurrent units)과 게이팅 메커니즘(gating mechanism)을 사용하여, 현재 위치의 특징 값을 업데이트할 때 이전 위치의 정보와 현재 제안된 값의 기여도를 조절하며 전역적인 의존성을 캡처한다.

**다. Project-out Stage**
Correction 단계에서 생성된 특징 맵을 다시 원래 입력 표현의 채널 수와 동일하게 매핑하여 출력한다.

### 2. SDU-Net 및 SDNU-Net 아키텍처

제안된 아키텍처는 기존 U-Net 및 U-Net++의 설계를 유지하면서 특정 위치에 SDN 레이어를 통합한 형태이다.

* **SDU-Net**: 표준 U-Net 구조에 SDN 레이어를 통합한 버전으로, 연산 속도가 빠르고 메모리 소비가 적은 효율적인 모델이다.
* **SDNU-Net**: U-Net++의 중첩 구조(nested architecture)에 SDN 레이어를 통합한 버전으로, 더 높은 정확도를 목표로 설계되었다.

두 모델 모두 동일한 깊이 레벨에서 Convolutional layer와 Spatial Dependency layer를 연속적으로 적용한다. 이를 통해 국소적인 유사성 비교(CNN)와 비국소적인 유사성 비교(SDN)를 동시에 수행하여 특징 맵의 지역적 일관성을 높이고 장거리 상호작용을 강화한다. 특히 SDNU-Net의 경우 연산 복잡도를 줄이기 위해 U-Net++의 하위 스케일(lower scales)에 SDN 레이어를 배치하는 전략을 사용하였다.

## 📊 Results

### 실험 설정

* **데이터셋**:
    1. Nuclei (현미경 이미지, 670장)
    2. Polyps (결장내시경 비디오, 612프레임)
    3. Liver (복부 CT 스캔, 131 볼륨)
* **평가 지표**: Dice score, Jaccard score
* **비교 대상**: Baseline U-Net, U-Net++
* **학습 설정**: Dice loss와 Cross-entropy loss의 조합을 사용하였으며, Early-stopping을 통해 최적 모델을 선택하였다.

### 정량적 결과

실험 결과, SDN을 도입한 모델들이 모든 태스크에서 베이스라인보다 우수한 성능을 보였다. (Table 2 참조)

| 모델 | Nuclei (Dice) | Polyps (Dice) | Liver (Dice) |
| :--- | :---: | :---: | :---: |
| U-Net | $91.79 \pm 0.32$ | $75.37 \pm 0.60$ | $77.44 \pm 2.25$ |
| U-Net++ | $92.64 \pm 0.24$ | $76.33 \pm 1.69$ | $80.24 \pm 2.10$ |
| **SDU-Net** | $\mathbf{93.25 \pm 0.35}$ | $\mathbf{80.62 \pm 0.79}$ | $\mathbf{82.43 \pm 2.24}$ |
| **SDNU-Net** | $\mathbf{94.10 \pm 0.36}$ | $\mathbf{83.31 \pm 1.30}$ | $\mathbf{85.72 \pm 2.45}$ |

전체적으로 Dice score 기준 평균 4.29포인트의 향상이 있었으며, 특히 폴립(Polyp) 분할($+6.98$)과 간(Liver) 분할($+5.48$)에서 SDNU-Net이 U-Net++ 대비 비약적인 성능 향상을 보였다. 이는 폴립과 간의 경우 크기가 다양하고 정상 조직과의 텍스처 차이가 커서 전역적 맥락 파악이 특히 중요하기 때문으로 분석된다.

### 소거 연구 (Ablation Study)

SDN 관련 하이퍼파라미터 분석 결과 다음과 같은 사실이 확인되었다.

* SDN 레이어를 적용하는 스케일의 수, 스캔 방향의 수, Project-in 단계의 출력 채널 수가 증가할수록 전반적인 성능이 향상되었다.
* 스캔 방향의 경우, 한 축으로 양방향 스캔을 하는 것보다 서로 다른 두 축(예: 가로축 하나, 세로축 하나)에 대해 스캔을 수행하는 것이 더 효과적이었다.

## 🧠 Insights & Discussion

본 연구의 강점은 CNN의 고질적인 문제인 제한된 수용 영역 문제를 SDN이라는 명시적인 구조를 통해 해결했다는 점이다. 특히 정성적 분석(Fig 1)에서 알 수 있듯이, 기존 U-Net 계열은 국소적 특징만으로 판단하여 엉뚱한 곳을 ROI로 지정하는 경우가 많았으나, SDU-Net 계열은 이미지 전체를 조망하여 단일한 ROI를 정확하게 식별해내는 능력을 보여주었다.

다만, SDN의 도입은 연산 비용과 메모리 사용량의 증가를 수반한다. 논문에서는 이를 완화하기 위해 SDN 레이어를 모든 층이 아닌 하위 스케일에만 전략적으로 배치하였다. 그럼에도 불구하고 SDNU-Net의 파라미터 수와 연산량(Gmac)이 베이스라인보다 증가한 점은 실제 실시간 진단 시스템 적용 시 고려해야 할 트레이드-오프 관계이다.

또한, 본 논문은 세 가지 서로 다른 모달리티(현미경, 내시경, CT)에서 일관된 성능 향상을 입증함으로써, 제안하는 방법론이 특정 데이터셋에 국한되지 않고 의료 영상 분할 전반에 걸쳐 일반화될 가능성이 높음을 시사한다.

## 📌 TL;DR

본 논문은 CNN 기반 의료 영상 분할 모델의 제한된 수용 영역 문제를 해결하기 위해, 무제한 수용 영역을 가진 **Spatial Dependency Networks (SDN)**를 U-Net 및 U-Net++에 통합한 **SDU-Net**과 **SDNU-Net**을 제안한다. 이 아키텍처들은 이미지의 전역적 공간 일관성을 명시적으로 모델링하여, 특히 크기와 형태 변화가 심한 폴립 및 간 분할 태스크에서 기존 모델 대비 월등한 성능 향상을 달성하였다. 이는 향후 고정밀 자동 의료 영상 진단 시스템 구축에 있어 중요한 아키텍처적 대안이 될 것으로 보인다.
