# Co-Learning Semantic-aware Unsupervised Segmentation for Pathological Image Registration

Yang Liu and Shi Gu (2025)

## 🧩 Problem to Solve

본 논문은 병변(lesion)이 존재하는 병리적 이미지(pathological image)의 등록(registration) 문제를 해결하고자 한다. 일반적인 변형 가능한 이미지 등록(deformable image registration)은 두 이미지 사이에 공간적 대응 관계가 존재한다는 가정하에 유사도 점수를 최대화하는 방향으로 수행된다. 그러나 뇌종양과 같은 병리적 이미지의 경우, 병변 부위로 인해 공간적 대응 정보가 손실되거나 조직의 비정상적인 왜곡이 발생하여 기존 방식으로는 큰 등록 오차가 발생한다.

이 문제의 중요성은 의료 영상 분석에서 병리적 이미지를 표준 아틀라스(atlas)에 정렬하거나, 수술 전후의 영상을 비교하는 longitudinal registration 작업이 필수적이라는 점에 있다. 따라서 본 논문의 목표는 레이블이 없는 상태에서 병변 부위를 자동으로 식별하고, 이를 처리하여 병리적 이미지에서도 정확한 등록을 달성할 수 있는 완전히 비지도 학습(completely unsupervised learning) 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Generation, Inpainting, and Registration (GIR)** 원칙을 결합한 **GIRNet**이라는 tri-net 협력 학습(collaborative learning) 프레임워크를 제안하는 것이다.

중심적인 설계 직관은 등록(Registration), 분할(Segmentation), 인페인팅(Inpainting) 세 가지 모듈을 동시에 학습시켜 서로가 서로의 성능을 보완하도록 만드는 것이다. 구체적으로, 등록 네트워크가 제공하는 시맨틱 정보(semantic information)를 이용해 병변 부위를 정확히 분할하고, 분할된 마스크를 이용해 병변 부위를 정상 조직으로 복원(inpainting)하며, 이렇게 복원된 이미지를 다시 등록 네트워크의 입력으로 사용하여 등록 정확도를 높이는 상호 보완적 루프를 구성한다.

## 📎 Related Works

기존의 병리적 이미지 등록 접근 방식은 크게 세 가지로 분류된다:
1. **Cost function masking**: 비대응 영역을 분할하여 유사도 측정 시 해당 영역을 마스킹하는 방법이다.
2. **Converting pathological image to normal appearance**: 생성 모델이나 저차원-희소 행렬 분해를 통해 병변 부위를 정상 조직으로 재구성하는 방법이다.
3. **Non-correspondence detection via intensity criteria**: 강도(intensity) 기준을 통해 등록 과정에서 비대응 영역을 동시에 검출하는 방법이다.

이러한 기존 방식들의 한계는 등록 과정에서 정답(ground truth)이나 매우 정확한 레이블이 필요하거나, 병변 부위가 클 경우 정렬 정확도가 떨어진다는 점이다. 또한 강도 기반 검출 방식은 데이터셋에 매우 민감하며 통합된 하이퍼파라미터를 찾기 어렵다는 단점이 있다. GIRNet은 데이터 독립적인 분할 모듈과 모달리티 적응형 인페인팅 모듈을 등록 파이프라인에 통합함으로써 이러한 한계를 극복하고자 한다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인
GIRNet은 $\text{RegNet}(\psi)$, $\text{SegNet}(\theta)$, $\text{InpNet}(\phi)$의 세 가지 네트워크로 구성되며, 이들은 co-learning 방식으로 동시에 업데이트된다.

1. **RegNet**: 소스 이미지 $S$와 템플릿 이미지 $T$를 입력받아 $S \rightarrow T$ 및 $T \rightarrow S$로의 변형 필드(deformation field) $\phi_{ST}$와 $\phi_{TS}$를 생성한다.
2. **InpNet**: $\text{SegNet}$이 생성한 마스크로 잘려나간 배경/전경 이미지와 $\text{RegNet}$에 의해 워핑된 이미지 $T \circ \phi_{TS}$를 입력받아 정상 외관의 전경/배경 이미지를 출력한다.
3. **SegNet**: 병리 이미지 $S$를 입력받고, $\text{InpNet}$이 복원한 정상 전경/배경 이미지를 참고하여 Minimal Mutual Information (MMI) 기반으로 병변 영역을 분할한다.

### 주요 구성 요소 및 손실 함수

#### 1. RegNet (Registration Network)
등록 오차를 줄이기 위해 병변 영역을 제외한 정상 영역에서만 손실 함수를 계산한다. $\text{SegNet}$의 출력 $\theta(S)$를 이용하여 병변 부위를 마스킹하고 인페인팅된 이미지와 템플릿 이미지 간의 유사도를 측정한다.
$$\text{L}_{reg} = \min_{\psi} \left( \text{L}_{sym}(\phi(S \cdot \theta(S) | T \circ \phi_{TS}) \circ \phi_{ST}, T) + \text{L}_{sym}(T \circ \phi_{TS}, \phi(S \cdot \theta(S) | T \circ \phi_{TS})) \right)$$
여기서 $\text{L}_{sym}$은 방향 일관성, 정규화 및 크기를 균형 있게 조절하는 SymNet의 손실 함수를 사용한다.

#### 2. SegNet (Segmentation Network)
비지도 분할을 위해 MMI(Minimal Mutual Information)를 사용한다. 단순한 MMI는 병변과 정상 조직의 강도가 비슷할 때 성능이 떨어지므로, $\text{RegNet}$의 변형 필드 $\phi_{TS}$를 이용해 건강한 이미지 $T$를 $S$로 워핑함으로써 시맨틱 정보를 주입한다. 분할 손실 함수는 다음과 같은 적대적 최적화 형태로 정의된다:
$$\text{L}_{seg} = \max_{\theta} \min_{\phi} \left( \frac{\mathbb{E}\{\theta(S) \cdot \text{D}[S, \phi(S \cdot \theta(S) | T \circ \phi_{TS})] \}}{\mathbb{E}\|\theta(S)\|} - \frac{\mathbb{E}\{\overline{\theta(S)} \cdot \text{D}[S, \phi(S \cdot \theta(S) | T \circ \phi_{TS})] \}}{\mathbb{E}\overline{\theta(S)}} \right)$$
여기서 $\text{D}$는 LNCC(Localized Normalized Cross-Correlation) 거리 함수이며, $\text{SegNet}$은 병변(전경)과 정상 조직(배경) 사이의 거리를 최대화하려 하고, $\text{InpNet}$은 이를 최소화하려 한다.

#### 3. InpNet (Inpainting Network)
인페인팅 네트워크는 상호 정보량 손실($\text{L}_{MI}$)과 유사도 손실($\text{L}_{sim}$)의 조합으로 학습된다.
$$\text{L}_{inp} = \text{L}_{MI} + \lambda \text{L}_{sim}$$
- $\text{L}_{MI}$: NCC(Normalized Correlation Coefficient)를 사용하여 원본과 복원 이미지 간의 상관관계를 최적화한다.
- $\text{L}_{sim}$: MSE(Mean Squared Error)를 사용하여 복원된 조직이 건강한 조직의 특성을 갖도록 유도한다.
또한, 도메인 차이를 줄이기 위해 히스토그램 매칭(Histogram Matching, HM)을 적용하여 $T \circ \phi_{TS}$의 히스토그램을 $S$와 유사하게 변환한다.

## 📊 Results

### 실험 설정
- **데이터셋**: OASIS-1 (알츠하이머), BraTS2020 (교모세포종), BraTS-Reg 2022 (랜드마크 포함), 그리고 이를 조합한 Pseudo 데이터셋을 사용하였다.
- **작업**: 아틀라스 기반 등록(Atlas-based registration), Longitudinal registration, 비지도 분할(Unsupervised segmentation).
- **지표**: MDE(Mean Deformation Error), TRE(Target Registration Error), DSC(Dice Similarity Coefficient).

### 주요 결과
1. **아틀라스 기반 등록**: Pseudo 데이터셋에서 MDE를 측정했을 때, 히스토그램 매칭(HM)을 적용한 GIRNet이 모든 영역(병변 내부, 근처, 멀리 떨어진 영역)에서 가장 우수한 성능을 보였다. 특히 정상 영역에서는 MDE가 $1\text{mm}$ 미만으로 기록되었다.
2. **Longitudinal registration**: pre-operative 영상을 follow-up 영상에 등록하는 작업에서 TRE를 측정하였다. RegNet을 CIR-DM으로 교체한 $\text{GIR(CIRDM)}$ 모델은 최신 기법인 DIRAC과 대등한 수준의 성능을 달성하였다.
3. **비지도 분할**: Pseudo 데이터셋에서 DSC $0.83$을 기록하며 AUCseg, NCRNet, DIRAC보다 높은 성능을 보였다. BraTS2020 데이터셋에서도 post-processing을 통해 $0.611$의 DSC를 달성하였다.

## 🧠 Insights & Discussion

본 연구는 등록, 분할, 인페인팅이라는 세 가지 서로 다른 작업을 하나의 협력 학습 프레임워크로 묶어, 레이블 없이도 병리적 이미지의 정렬 문제를 해결할 수 있음을 보여주었다. 특히 $\text{RegNet}$이 제공하는 공간적 워핑 정보가 $\text{SegNet}$의 비지도 분할 성능을 끌어올리는 결정적인 역할을 한다는 점이 확인되었다.

**강점 및 한계**:
- **강점**: 완전히 비지도 학습 방식으로 구현되어 레이블링 비용이 들지 않으며, 도메인 차이를 극복하기 위한 히스토그램 매칭의 효과를 입증하였다.
- **한계**: $\text{InpNet}$의 복원 품질이 $\text{SegNet}$의 분할 정확도에 직접적인 영향을 미치고, 이것이 다시 등록 결과로 이어지는 의존 구조를 가진다. 따라서 인페인팅 성능의 한계가 전체 시스템의 병목 지점이 될 수 있다.
- **비판적 해석**: 논문에서 제시된 Pseudo 데이터셋은 실제 임상 데이터보다 단순화된 환경일 가능성이 크다. 실제 데이터셋(BraTS2020)에서의 DSC가 Pseudo 데이터셋보다 낮게 나타나는 점은 실제 병변의 복잡성이 인페인팅과 분할 과정에 여전히 큰 도전 과제임을 시사한다.

## 📌 TL;DR

GIRNet은 등록(RegNet), 분할(SegNet), 인페인팅(InpNet) 모듈이 서로의 정보를 교환하며 학습하는 비지도 협력 학습 프레임워크이다. MMI 기반의 시맨틱-인식 분할과 히스토그램 매칭을 통한 인페인팅을 통해, 병변으로 인한 공간적 대응 손실 문제를 해결하고 정밀한 병리적 이미지 등록을 가능하게 한다. 이 연구는 레이블 없이도 병변 검출과 이미지 정렬을 동시에 수행할 수 있는 효율적인 경로를 제시하여 향후 의료 영상 분석의 비용 절감 및 자동화에 기여할 가능성이 높다.