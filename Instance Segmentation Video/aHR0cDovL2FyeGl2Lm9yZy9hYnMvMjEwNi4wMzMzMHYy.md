# Contextual Guided Segmentation Framework for Semi-supervised Video Instance Segmentation

Trung-Nghia Le, Tam V. Nguyen, Minh-Triet Tran (2022)

## 🧩 Problem to Solve

본 논문은 **Semi-supervised Video Instance Segmentation (VOS)** 문제를 해결하고자 한다. 이 작업은 비디오의 첫 번째 프레임에서 주어진 특정 인스턴스의 Ground-truth 마스크를 바탕으로, 이후의 모든 프레임에서 해당 인스턴스들을 정확하게 분할하고 일관된 ID를 부여하는 것을 목표로 한다.

비디오 인스턴스 분할은 단순한 객체 분할보다 훨씬 난이도가 높은데, 이는 비디오 데이터가 가진 다음과 같은 특성들 때문이다:
- **급격한 움직임(Rapid motion)** 및 **빠른 변형(Large deformations)**
- **폐색(Occlusions)** 및 **복잡한 객체 간 상호작용(Complex object interactions)**
- **방해 요소(Distractors)**, **작은 객체**, 그리고 **세밀한 구조(Fine structures)**

기존 연구들은 추적 및 재식별(Re-identification) 방법을 통합하여 일관성을 유지하려 했으나, 비디오 내의 다양한 컨텍스트(Context)를 모두 포괄하지 못해 분할에 실패하는 경우가 많았다. 따라서 본 논문은 **컨텍스트 정보(Context information)**를 적극적으로 활용하여 모호성을 줄이고 강건한(Robust) 분할 결과를 얻는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 "두 번 확인하라(Look twice)"는 직관에서 영감을 얻은 **Contextual Guided Segmentation (CGS)** 프레임워크이다. 이 프레임워크는 총 3단계의 패스(Pass)를 통해 단계적으로 분할 정밀도를 높인다.

1.  **Instance Re-Identification Flow (IRIF)**: 첫 번째 패스(Preview Segmentation)에서 인스턴스의 속성(인간 여부, 강체/변형 가능 여부, 알려진 카테고리 여부)을 분석하여 최적의 분할 전략을 선택할 수 있는 기반을 마련한다.
2.  **Contextual Segmentation**: 두 번째 패스에서 분석된 속성에 따라 맞춤형 분할 기법을 적용한다. (예: 인간은 스켈레톤 가이드, 강체는 합성 데이터 기반 FCN 적용)
3.  **Guided Fine-grained Segmentation**: 세 번째 패스에서 직사각형 형태의 ROI가 아닌, 주변 프레임의 어텐션을 이용한 **비직사각형 ROI(Non-rectangular ROI)**를 구축하여 배경 노이즈를 제거하고 세밀한 경계를 추출한다.
4.  **Wonderland Data**: One-shot learning의 한계를 극복하기 위해, Places365 데이터셋의 실제 배경을 활용하여 기존 Lucid Data보다 훨씬 방대하고 현실적인 합성 학습 데이터를 생성한다.

## 📎 Related Works

논문에서는 VOS와 관련된 기존 연구를 세 가지 범주로 분류하여 설명한다.

1.  **One-Shot Learning**: 첫 프레임만으로 학습하는 방식으로, 데이터 증강이 필수적이다. Lucid Dreaming과 같은 기법이 사용되었으나, 이는 배경 변화를 충분히 다루지 못한다는 한계가 있다. 본 논문은 이를 개선한 Wonderland Data를 제안한다.
2.  **Temporal Connection Mining**: 인스턴스 추적, 전파 및 재식별을 통해 프레임 간 연결성을 찾는 방식이다. Mask propagation과 Re-identification feature embedding 등을 사용하여 소실된 객체를 복구하려 한다.
3.  **End-to-End Temporal Learning**: LSTM, Memory Network, Guided-attention 등을 통해 시간적 정보를 직접 학습하는 방식이다. 최근에는 STM(Space-Time Memory) 네트워크 등을 통해 특징 맵을 결합하는 방식이 주를 이룬다.

본 연구는 이러한 기존 방식들이 다양한 컨텍스트를 충분히 반영하지 못한다는 점을 지적하며, 속성 기반의 맞춤형 전략과 정교한 ROI 가이드를 통해 차별화를 꾀한다.

## 🛠️ Methodology

CGS 프레임워크는 크게 세 가지 패스와 최종 병합 단계로 구성된다.

### 1. Preview Segmentation (First Pass)
이 단계에서는 **Instance Re-Identification Flow (IRIF)**를 통해 각 인스턴스의 컨텍스트 속성을 추출한다.
- **Localization & Tracking**: 인간 객체는 Faster R-CNN 기반의 Person Search를 사용하고, 비인간 객체는 DeepFlow와 DPM(Deformable Part Models)을 사용하여 추적한다.
- **Adaptive Online Learning**: 이전 $n$개 프레임의 외형을 학습한 다중 이진 SVM(Binary SVM) 분류기와 GrabCut을 사용하여 인스턴스를 분할한다.
- **Contextual Property Extraction**: Mask R-CNN을 통해 카테고리를 파악하고, 첫 $n_{Preview}$ 프레임 동안 호모그래피 행렬(Homography matrix)의 존재 여부를 분석하여 해당 객체가 **강체(Rigid)**인지 **변형 가능(Deformable)**한지 판별한다.

### 2. Contextual Segmentation (Second Pass)
추출된 속성에 따라 서로 다른 분할 스킴을 적용한다.
- **Human Instance**: Mask R-CNN을 기본으로 하되, 특이한 포즈나 폐색 문제를 해결하기 위해 **OpenPose**의 스켈레톤 정보를 이용한 스켈레톤 가이드 분할을 수행한다. 이후 Object Flow를 통해 프레임 간 일관성을 보정한다.
- **Rigid Non-Human**: **Wonderland Data**로 학습된 DeepLab2와 OSVOS를 사용한다. Wonderland Data는 Places365 데이터셋에서 비디오 장면과 유사한 배경을 검색하여 합성한 데이터셋으로, 기존 Lucid Data보다 훨씬 많은 10,000장의 이미지를 생성한다.
- **Deformable Non-Human**: 알려진 카테고리는 Mask R-CNN을 사용하고, 알려지지 않은 카테고리는 IRIF의 결과를 그대로 사용한다.

### 3. Guided Segmentation (Third Pass)
전통적인 직사각형 ROI의 한계를 극복하기 위해 **비직사각형 ROI** 기반의 세밀한 분할을 수행한다.
- **Bi-directional Propagation**: 
    - **Forward Propagation**: 이전 프레임의 마스크를 참조하여 ROI를 생성함으로써, 객체가 밀집된 지역에서 과도하게 분할되는 것을 방지한다.
    - **Backward Propagation**: 이후 프레임의 마스크를 참조하여 빠른 움직임이나 폐색으로 인해 사라졌던 인스턴스를 복구한다.
- **Non-Rectangular ROI Construction**: 주변 프레임의 마스크를 확장하고 현재 프레임으로 전송하여 결합함으로써, 객체의 실제 형태에 가까운 ROI를 생성한다. 이때 경계면의 급격한 변화를 막기 위해 Blur 마스크를 적용한 Smooth transition region을 구축한다.
- **Fine-grained Segmentation**: 생성된 가이드 ROI 내에서 Deep Grabcut, Mask R-CNN, 그리고 Xception-65 백본 기반의 DeepLab3+를 사용하여 최종 정밀 분할을 수행한다.

### 4. Refinement and Merging
- **Refinement**: 면적이 전체의 5% 미만인 **희귀 인스턴스(Rare instance)**에 대해 binary-SVM의 foreground 확률을 이용하여 복구하고, Boundary Snapping 기법으로 경계를 정밀화한다.
- **Merging**: 여러 인스턴스가 겹칠 때의 우선순위를 결정하기 위해 다음의 **Topological Order(Z-order)**를 적용한다:
    1. **상호작용 휴리스틱**: (멀리 있음) 운송수단 $\rightarrow$ (중간) 인간 $\rightarrow$ (가까움) 손에 든 작은 객체.
    2. **깊이 값(Depth values)**: DCNF-FCSP를 통해 추정된 픽셀별 깊이의 평균값을 사용한다.
    3. **희귀 인스턴스 우선순위**: 크기가 작은 희귀 객체가 카메라에 더 가깝다고 가정한다.

## 📊 Results

### 실험 설정
- **데이터셋**: DAVIS Test-Challenge (150개 시퀀스, 10,459개 프레임)
- **평가 지표**: Region Jaccard index ($\text{J}$), Boundary F-measure ($\text{F}$), 그리고 이 둘의 평균인 Global Score를 사용한다.

### 주요 결과
- **DAVIS Challenge 성적**: 본 방법론은 2017년 3위, 2018년 6위, 2019년 3위를 기록하며 매우 경쟁력 있는 성능을 보였다.
- **정량적 수치 (2019년 기준)**:
    - Global Score: $75.4\%$
    - Region Similarity ($\text{J}$): $72.4\%$
    - Contour Accuracy ($\text{F}$): $78.4\%$
- 특히 $\text{J}$ decay와 $\text{F}$ decay 지표에서 최상위권 성능을 유지하여, 비디오가 진행됨에 따라 성능이 급격히 떨어지지 않는 **안정성(Stability)**을 입증하였다.

### Ablation Study
- **Pass별 기여도**:
    - Preview Segmentation (PS)만 사용 시: Global Score $63.8\%$
    - PS + Contextual Segmentation (CS) 사용 시: Global Score $66.3\%$ (CS가 약 $2.5\%$ 향상)
    - PS + CS + Guided Segmentation (GS) 모두 사용 시: Global Score $75.4\%$ (GS가 약 $9.1\%$ 추가 향상)
- 이를 통해 세 가지 패스를 모두 거치는 CGS 프레임워크의 유효성이 증명되었다.

## 🧠 Insights & Discussion

**강점**
- 본 논문은 단일 모델에 의존하지 않고, 객체의 속성에 따라 최적의 알고리즘을 선택적으로 적용하는 **전략적 파이프라인**을 구축하였다.
- 특히 비직사각형 ROI와 양방향 전파(Bi-directional propagation)를 통해 VOS의 고질적인 문제인 폐색과 급격한 외형 변화 문제를 효과적으로 해결하였다.
- 합성 데이터 생성 시 실제 배경 데이터셋(Places365)을 활용하여 One-shot learning의 일반화 성능을 크게 높인 점이 인상적이다.

**한계 및 논의사항**
- 매우 복잡한 파이프라인으로 인해 추론 속도가 느릴 가능성이 높다. 각 패스를 순차적으로 수행하고 여러 외부 모델(OpenPose, Mask R-CNN, DeepLab 등)을 호출하므로 실시간 적용에는 어려움이 있을 것으로 보인다.
- 인스턴스 간의 상호작용을 휴리스틱(Heuristics) 기반의 Z-order로 해결하였는데, 이를 딥러닝 기반의 관계 모델링으로 대체한다면 더 정교한 병합이 가능할 것이다.
- 논문에서 언급되었듯이, 위장(Camouflage) 분석이나 캡슐 네트워크(Capsule Network) 같은 최신 구조의 도입이 향후 성능 향상의 열쇠가 될 것으로 보인다.

## 📌 TL;DR

본 논문은 Semi-supervised VOS를 위해 **Preview $\rightarrow$ Contextual $\rightarrow$ Guided**라는 3단계의 정밀 분할 프레임워크(CGS)를 제안한다. 객체의 속성을 먼저 분석하고, 그에 맞는 맞춤형 분할 기법을 적용하며, 마지막으로 비직사각형 ROI와 양방향 전파를 통해 세밀한 경계를 완성한다. DAVIS Challenge에서 일관되게 상위 3~6위권의 성적을 거두었으며, 특히 합성 데이터 증강(Wonderland Data)과 가이드 기반 분할을 통해 높은 안정성과 정확도를 달성하였다. 이 연구는 복잡한 비디오 환경에서 컨텍스트 정보를 어떻게 단계적으로 활용해야 하는지에 대한 중요한 방법론을 제시한다.