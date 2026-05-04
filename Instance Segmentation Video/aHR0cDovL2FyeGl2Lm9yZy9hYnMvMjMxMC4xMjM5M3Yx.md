# Deep Learning Techniques for Video Instance Segmentation: A Survey

Chenhao Xu, Chang-Tsun Li, Yongjian Hu, Chee Peng Lim, Douglas Creighton (2023)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 비디오 내의 개별 객체 인스턴스를 동시에 탐지(Detection), 분할(Segmentation), 그리고 추적(Tracking)하는 **Video Instance Segmentation (VIS)** 기술의 전반적인 분석과 체계화이다.

VIS는 이미지 단위의 인스턴스 분할을 넘어 시간축에 따른 인스턴스의 일관성을 유지해야 한다는 점에서 매우 까다로운 과제이다. 이 기술은 자율주행 자동차의 보행자 및 차량 추적, 의료 영상 분석, 보안 감시 시스템의 행동 인식 등 실세계의 다양한 응용 분야에서 핵심적인 역할을 수행한다. 따라서 본 논문의 목표는 급격히 증가하는 VIS 관련 딥러닝 기법들을 아키텍처 관점에서 분류하고, 성능, 복잡도, 계산 비용을 비교 분석하며, 향후 연구 방향을 제시하는 종합적인 가이드를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 주요 기여 사항은 다음과 같다.

1. **아키텍처 기반의 체계적 분류**: VIS 딥러닝 모델을 특징 처리 방식에 따라 Multi-stage, Multi-branch, Hybrid, Integrated, Recurrent의 다섯 가지 범주로 정의하고 정성적으로 비교하였다.
2. **보조 기술(Auxiliary Techniques) 분석**: 모델 성능을 향상시키기 위한 데이터셋의 특성과 표현 학습(Representation Learning) 방법론을 정리하였다.
3. **미해결 과제 및 미래 방향 제시**: 폐색(Occlusion), 모션 블러(Motion Blur), 어노테이션 효율성, 오픈 보캐블러리(Open-Vocabulary) 등 VIS 분야가 직면한 주요 도전 과제들을 식별하고 연구 방향을 제안하였다.

## 📎 Related Works

논문에서는 VIS를 이해하기 위해 먼저 비디오 분할의 세 가지 하위 작업(VOS, VSS, VIS)을 정의하며 기존 연구와의 차별점을 설명한다.

- **Video Object Segmentation (VOS)**: 전경 객체를 배경으로부터 분리하는 이진 분할 작업이다. 동일 클래스의 서로 다른 인스턴스를 구분하지 않는다.
- **Video Semantic Segmentation (VSS)**: 모든 픽셀을 특정 클래스로 분류하지만, 개별 인스턴스를 구분하는 단계는 없다.
- **Video Instance Segmentation (VIS)**: 각 프레임에서 인스턴스를 분할함과 동시에, 프레임 간에 동일한 인스턴스에 동일한 ID를 부여하여 추적하는 작업이다. 이는 Multi-Object Tracking and Segmentation (MOTS)와 매우 유사하며 본 논문에서는 혼용하여 사용한다.

기존의 서베이 논문들이 주로 이미지 분할이나 단순 객체 추적(MOT)에 집중했던 것과 달리, 본 논문은 '분할'과 '추적'이 통합된 VIS 영역만을 집중적으로 다룬다는 점에서 차별성을 가진다.

## 🛠️ Methodology

본 논문은 VIS 아키텍처를 '넥(Neck)' 단계에서의 특징 처리 방식에 따라 다음과 같이 분류하여 상세히 설명한다.

### 1. Multi-Stage Feature Processing

특징 추출과 변환 단계를 순차적으로 진행하는 구조이다.

- **절차**: 프레임 수준 특징 추출 $\rightarrow$ 관심 영역(RoI) 제안 $\rightarrow$ 객체 분류 및 마스크 생성 $\rightarrow$ 프레임 간 인스턴스 연관(Association) 및 추적.
- **대표 모델**: MaskTrack R-CNN, TrackR-CNN.
- **특징**: 저수준 및 고수준 특징 추출에 효과적이지만, 단계가 많아질수록 계산 복잡도가 증가한다.

### 2. Multi-Branch Feature Processing

여러 브랜치가 병렬로 작동하여 서로 다른 측면의 표현을 학습하는 구조이다.

- **절차**: 각 브랜치가 서로 다른 하위 작업(예: 시맨틱 분할 vs 살리언스 객체 분할, 혹은 탐지 vs 추적)을 수행하고, 그 결과를 융합하여 최종 VIS 결과를 도출한다.
- **대표 모델**: SISO, YOLACT 계열, Siamese Network 기반 모델.
- **특징**: 상호 보완적인 정보를 캡처하여 견고한 표현 학습이 가능하지만, 파라미터 수가 많고 브랜치 간의 균형을 맞추는 튜닝이 어렵다.

### 3. Hybrid Feature Processing

Multi-stage와 Multi-branch의 장점을 결합한 형태이다.

- **절차**: 각 브랜치 내부에서 Multi-stage 처리를 수행하여 높은 시맨틱 수준의 특징을 얻고, 동시에 브랜치 간의 병렬 처리를 통해 판별력을 높인다.
- **특징**: 가장 강력한 성능을 낼 수 있으나, 설계가 매우 복잡하고 계산 비용이 매우 높다.

### 4. Integrated Feature Processing

비디오나 클립 전체를 하나의 3D 시공간 볼륨으로 처리하는 구조이다.

- **절차**: 전 프레임의 특징을 한꺼번에 추출하여 3D 시공간 특징 맵을 구축하고, 이를 인코더-디코더 구조(주로 Transformer)를 통해 처리하여 한 번에 마스크 시퀀스를 예측한다.
- **대표 모델**: VisTR, SeqFormer, Mask2Former (VIS 확장판).
- **특징**: 전역적인 맥락 파악에 유리하며 구조가 우아하지만, 방대한 데이터와 메모리, 긴 학습 시간이 필요하다.

### 5. Recurrent Feature Processing

시간축을 따라 특징을 순차적으로 전파하는 구조이다.

- **절차**: 이전 프레임의 특징이나 쿼리(Query)를 현재 프레임으로 전달하여 인스턴스를 추적한다.
- **대표 모델**: ConvLSTM 기반 모델, TrackFormer (Query Propagation).
- **특징**: 메모리 오버헤드가 적고 실시간 처리에 유리하지만, 매우 긴 비디오의 경우 장기 의존성(Long-term dependency) 문제로 인해 성능이 저하될 수 있다.

## 📊 Results

본 논문은 개별 실험 결과보다는 기존 문헌들의 성과를 종합적으로 분석한 결과를 제시한다.

### 데이터셋 및 지표

VIS 성능 평가를 위해 가장 널리 사용되는 데이터셋은 **YouTube-VIS**이며, 최근에는 폐색 문제가 강조된 **OVIS**, 주행 환경의 **NuImages** 등이 사용되고 있다. 주요 평가지표로는 $\text{AP}$ (Average Precision) 등이 활용된다.

### 아키텍처별 비교 요약

- **효율성**: Recurrent $\approx$ Multi-Branch (일부) $>$ Multi-Stage $>$ Integrated.
- **정확도 및 견고성**: Integrated $\approx$ Hybrid $>$ Multi-Stage $\approx$ Recurrent.
- **메모리 사용량**: Integrated $\gg$ Hybrid $>$ Multi-Stage $>$ Recurrent.

### 보조 기술의 효과

- **표현 학습**: Temporal Pyramid Routing (TPR)과 같은 기법이 클립 수준의 이해도를 높인다.
- **데이터 증강**: Continuous Copy-Paste (CCP) 기법이 학습 데이터 부족 문제를 완화하고 추적 성능을 향상시킨다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 논문은 파편화되어 있던 VIS 연구들을 '특징 처리 흐름'이라는 명확한 기준으로 분류함으로써, 연구자가 자신의 목적(실시간성 vs 정확도)에 맞는 아키텍처를 선택할 수 있는 체계적인 프레임워크를 제공하였다. 특히 Transformer 기반의 Integrated 구조로 패러다임이 전환되고 있음을 명확히 짚어냈다.

### 한계 및 비판적 해석

- **계산 비용의 무시**: Integrated 및 Hybrid 모델들이 높은 성능을 보이지만, 실제 엣지 디바이스나 실시간 시스템에 적용하기에는 메모리와 연산량 장벽이 너무 높다는 점이 지적된다.
- **데이터 의존성**: Transformer 기반 모델들이 대규모 데이터셋에 크게 의존하고 있어, 데이터 획득이 어려운 특수 도메인(의료, 산업 현장)에서의 적용 가능성에 대해서는 의문이 남는다.

### 향후 연구 방향

논문은 단순한 성능 개선을 넘어 다음과 같은 방향으로의 확장을 제안한다.

- **Amodal VIS**: 객체가 가려졌을 때 보이지 않는 부분까지 예측하는 기술.
- **Open-Vocabulary VIS**: 학습 단계에서 보지 못한 새로운 클래스의 객체도 분할/추적하는 기술.
- **Promptable VIS**: SAM(Segment Anything Model)과 같이 텍스트나 포인트 프롬프트를 통해 유연하게 객체를 지정하는 기술.

## 📌 TL;DR

본 논문은 비디오 인스턴스 분할(VIS) 분야의 딥러닝 기법들을 **Multi-stage, Multi-branch, Hybrid, Integrated, Recurrent**의 5가지 아키텍처로 체계화하여 분석한 종합 서베이 보고서이다. VIS의 핵심인 '분할'과 '추적'의 통합 과정을 상세히 다루었으며, 특히 최근의 Transformer 기반 통합 처리 방식의 부상을 강조한다. 이 연구는 향후 **오픈 보캐블러리(Open-Vocabulary)** 및 **프롬프트 기반 분할**과 같은 차세대 비디오 이해 연구의 이정표 역할을 할 것으로 기대된다.
