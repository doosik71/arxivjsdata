# Segment and Track Anything

Yangming Cheng, Liulei Li, Yuanyou Xu, Xiaodi Li, Zongxin Yang, Wenguan Wang, Yi Yang (2023)

## 🧩 Problem to Solve

본 논문은 비디오 내의 임의의 객체를 정밀하고 효율적으로 분할(Segmentation)하고 추적(Tracking)하는 문제를 해결하고자 한다. 비디오 객체 분할(Video Object Segmentation, VOS)은 드론 기술, 자율 주행, 의료 영상, 증강 현실(AR) 등 다양한 분야에서 필수적이지만, 기존의 연구들은 비지도 학습(Unsupervised), 준지도 학습(Semi-supervised), 상호작용 기반(Interactive), 언어 유도 기반(Language-induced) 등 세부 작업별로 파편화되어 있었다.

특히, 최근 등장한 Segment Anything Model (SAM)은 이미지 수준에서는 뛰어난 제로샷(Zero-shot) 성능과 유연한 프롬프트 기능을 제공하지만, 이를 비디오에 직접 적용할 경우 프레임 간의 시간적 일관성(Temporal Coherence)을 고려하지 못해 결과가 최적화되지 않는 문제가 발생한다. 또한, SAM은 시맨틱 라벨을 출력하지 않으며 텍스트 프롬프트의 효율성이 낮아, 고수준의 시맨틱 이해가 필요한 Referring Object Segmentation 작업 등을 수행하는 데 한계가 있다. 따라서 본 논문의 목표는 SAM의 강력한 이미지 분할 능력과 효율적인 비디오 추적 메커니즘을 결합하여, 멀티모달 상호작용이 가능하고 새로운 객체 출현에 대응할 수 있는 통합 비디오 분할 프레임워크인 SAM-Track을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지 분할의 기초 모델인 SAM, 효율적인 멀티 객체 추적 모델인 DeAOT, 그리고 오픈셋 객체 탐지 모델인 Grounding-DINO를 하나의 파이프라인으로 통합하는 것이다. 

중심적인 설계 직관은 SAM을 통해 비디오의 키프레임(Key-frame)에서 정밀한 초기 마스크(Reference Mask)를 생성하고, 이를 DeAOT의 입력으로 사용하여 이후 프레임으로 전파(Propagation)함으로써 시간적 일관성을 유지하는 것이다. 여기에 Grounding-DINO를 결합하여 텍스트 기반의 객체 선택 기능을 추가함으로써, 사용자가 자연어로 추적 대상을 지정할 수 있는 유연성을 확보하였다.

## 📎 Related Works

논문에서는 다음과 같은 세 가지 핵심 구성 요소를 소개하며, 이들의 결합을 통해 기존 방식의 한계를 극복한다.

1.  **DeAOT**: AOT 기반의 VOS 모델로, 고차원 임베딩 공간에서 식별 메커니즘(Identification mechanism)을 사용하여 여러 객체를 단일 객체 추적과 유사한 속도로 추적할 수 있다. 계층적 Gated Propagation Module (GPM)을 통해 객체 불가지론적(Object-agnostic) 정보와 객체 특이적(Object-specific) 임베딩을 분리하여 전파한다.
2.  **Segment Anything Model (SAM)**: 대규모 데이터셋 SA-1B로 학습된 모델로, 포인트, 박스, 텍스트 등 유연한 프롬프트를 통해 고품질의 마스크를 생성한다. 하지만 비디오의 시간적 연속성을 처리하는 기능이 없다.
3.  **Grounding-DINO**: 언어 정보를 통합한 오픈셋 객체 탐지기로, 텍스트 카테고리나 상세 묘사를 통해 대상 객체의 최소 외접 사각형(Minimum external rectangle)을 반환하는 Referring Object Detection을 수행한다.

기존 접근 방식들이 단일 상호작용 모드나 특정 데이터셋에 최적화된 것과 달리, SAM-Track은 이 세 모델을 통합하여 상호작용 모드(Interactive)와 자동 모드(Automatic)를 모두 지원하는 통합 프레임워크를 지향한다.

## 🛠️ Methodology

SAM-Track은 크게 세 가지 추적 모드로 구성된다.

### 1. Interactive Tracking Mode
사용자가 직접 추적할 객체를 지정하는 모드이다. 
- **파이프라인**: $\text{Grounding-DINO} \rightarrow \text{SAM} \rightarrow \text{DeAOT}$
- **상세 과정**: 
    - 사용자가 텍스트를 입력하면 Grounding-DINO가 해당 객체의 Bounding Box를 탐지한다.
    - SAM은 이 Box나 사용자의 클릭(Click) 입력을 프롬프트로 받아 정밀한 객체 마스크 $S$를 생성한다.
    - DeAOT는 이 마스크를 초기 참조 프레임으로 사용하여 GPM을 통해 이후 프레임으로 시각적 임베딩과 ID 임베딩을 전파하며 추적을 수행한다.

### 2. Automatic Tracking Mode
비디오 중간에 새롭게 등장하는 객체를 자동으로 탐지하고 추적하는 모드이다. 두 가지 방법이 제시된다.
- **Segment Everything**: SAM의 전체 분할 기능을 사용하여 키프레임의 모든 객체 마스크를 얻는다.
- **Object of Interest Segmentation**: 미리 정의된 텍스트 프롬프트(예: "person", "car")를 통해 Grounding-DINO와 SAM이 새로운 객체만 추출한다.

새로운 객체를 정의하기 위해 **Comparing Mask Results (CMR)** 기법을 사용한다. 이는 DeAOT의 추적 결과와 SAM의 분할 결과를 비교하여, SAM은 찾았지만 DeAOT는 추적하지 못하고 있는 영역을 새로운 객체로 간주하는 방식이다.
- 새로운 객체 마스크 $N$은 다음과 같이 계산된다.
$$N = T^0 * S$$
여기서 $T^0$는 DeAOT 추적 결과의 배경(Background) 영역이고, $S$는 SAM의 분할 결과이다.
- 특정 객체 $x$에 대해 SAM 결과의 크기를 $x_s$, 새로운 객체 마스크 $N$에서의 크기를 $x_n$이라 할 때, 다음 조건이 만족되면 새로운 객체로 정의한다.
$$\text{CMR}(x) = \begin{cases} 1, & \text{if } \frac{x_n}{x_s} > t \\ 0, & \text{else} \end{cases}$$
여기서 $t$는 새로운 객체 판단을 위한 최소 임계값이다.

### 3. Fusion Tracking Mode
상호작용 모드와 자동 모드를 동시에 사용하는 방식이다. 첫 프레임에서는 사용자가 원하는 객체를 상호작용으로 지정하고, 비디오 진행 중 새롭게 나타나는 객체는 자동 모드를 통해 추가하여 추적한다.

## 📊 Results

### 정량적 평가 (Quantitative Results)
DAVIS-2016 Val 및 DAVIS-2017 Test 벤치마크를 통해 성능을 검증하였다. SAM-Track은 R50-DeAOT-L 모델을 기반으로 추적을 수행하였으며, 초기 마스크는 클릭(Click) 방식으로 생성하였다.

| Dataset | Metric | SAM-Track (Ours) | R50-DeAOT-L | SwinB-DeAOT-L |
| :--- | :---: | :---: | :---: | :---: |
| **DAVIS-2016 Val** | AvgJ | 92.0% | 92.3% | 92.9% |
| | JF | 90.3% | 90.5% | 91.1% |
| **DAVIS-2017 Test** | AvgJ | 79.2% | 80.7% | 82.8% |
| | JF | 75.3% | 76.9% | 78.9% |

*분석: SAM-Track은 전문적인 Mask 기반 초기화 모델인 DeAOT-L과 비교했을 때 약간 낮은 성능을 보이지만, 단순한 '클릭'만으로 유사한 수준의 성능을 낸다는 점에서 상호작용 효율성이 매우 높음을 입증하였다.*

### 정성적 평가 및 응용 (Qualitative Results & Applications)
다양한 도메인에서 SAM-Track의 실용성을 확인하였다.
- **스포츠 분석**: "soccer players"라는 텍스트 프롬프트와 클릭을 통해 선수와 경기장을 동시에 추적한다.
- **의료 분야**: 학습 데이터가 부족한 희귀 세포나 장기를 클릭 한 번으로 제로샷 추적할 수 있다.
- **스마트 시티 및 자율 주행**: 끊임없이 새로운 차량과 보행자가 등장하는 환경에서 자동 추적 모드를 통해 다수의 객체를 실시간으로 추적한다.

## 🧠 Insights & Discussion

본 연구의 강점은 SAM의 범용적인 분할 능력과 DeAOT의 효율적인 추적 능력을 결합하여, 복잡한 재학습 없이도 다양한 도메인에 즉시 적용 가능한 'Unified Framework'를 구축했다는 점이다. 특히 CMR 메커니즘을 통해 새로운 객체가 등장할 때 기존 객체의 ID가 바뀌는 ID exchange 문제를 완화한 점이 돋보인다.

다만, 정량적 결과에서 보듯 SAM을 통한 초기화가 정교하게 제작된 Ground Truth Mask를 사용하는 것보다는 약간 성능이 낮게 나타난다. 이는 SAM의 제로샷 마스크가 완벽하지 않을 수 있음을 시사하며, 향후 SAM의 마스크를 비디오 도메인에 맞게 미세 조정(Fine-tuning)하거나 정제하는 과정이 추가된다면 더 높은 성능을 기대할 수 있을 것이다. 또한, $n$번째 프레임마다 자동 모드를 호출하는 주기 설정에 따른 연산량과 정확도의 트레이드-오프에 대한 분석은 본 논문에서 명시적으로 다뤄지지 않았다.

## 📌 TL;DR

SAM-Track은 **SAM(분할) + DeAOT(추적) + Grounding-DINO(탐지)**를 통합하여, 클릭/박스/텍스트 등 멀티모달 상호작용으로 비디오 내 어떤 객체든 추적할 수 있게 한 프레임워크이다. 특히 새로운 객체를 자동으로 탐지하는 CMR 메커니즘을 도입하여 자율 주행, 의료, 스포츠 분석 등 실시간 객체 변화가 심한 실제 응용 분야에 매우 높은 활용 가능성을 제시하였다.