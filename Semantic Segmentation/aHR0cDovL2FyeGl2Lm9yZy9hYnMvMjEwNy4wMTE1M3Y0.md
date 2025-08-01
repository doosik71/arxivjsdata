# A Survey on Deep Learning Technique for Video Segmentation

Tianfei Zhou, Fatih Porikli, David J. Crandall, Luc Van Gool, Wenguan Wang

## 🧩 Problem to Solve

비디오 프레임을 여러 세그먼트나 객체로 분할하는 비디오 분할(Video Segmentation)은 시각 효과, 자율 주행, 화상 회의 등 다양한 응용 분야에서 핵심적인 역할을 합니다. 최근 딥러닝 기술의 발전으로 비디오 분할 성능이 크게 향상되었지만, 기존 연구 조사는 딥러닝 시대 이전에 수행되었거나(예: [14], [15]), 비디오 전경/배경 분할(Video Foreground/Background Segmentation)과 같은 특정 분야에만 초점을 맞추고 있어(예: [16], [17]) 빠르게 발전하는 이 분야의 최신 동향을 총체적으로 반영하지 못하는 문제가 있습니다. 이 논문은 새로운 연구자들이 이 분야에 진입하는 데 도움을 주고자 딥러닝 기반 비디오 분할에 대한 포괄적인 최신 연구 조사를 제공하는 것을 목표로 합니다.

## ✨ Key Contributions

* 비디오 객체 분할(Video Object Segmentation, VOS)과 비디오 시맨틱 분할(Video Semantic Segmentation, VSS)이라는 두 가지 주요 연구 분야를 포괄적으로 검토합니다.
* 각 분야의 작업 설정, 배경 개념, 개발 역사 및 주요 과제를 상세히 소개합니다.
* 대표적인 딥러닝 기반 방법론과 데이터셋에 대한 심층적인 개요를 제공합니다.
* 검토된 방법론들을 여러 잘 알려진 데이터셋에서 벤치마킹하여 성능을 비교 분석합니다.
* 이 분야의 미해결 과제(open issues)를 지적하고, 향후 연구 기회를 제안합니다.
* 연구 진행 상황을 지속적으로 추적할 수 있는 공개 웹사이트(<https://github.com/tfzhou/VS-Survey)를> 제공합니다.

## 📎 Related Works

본 논문은 비디오 분할의 역사를 전통적인 컴퓨터 비전 및 머신러닝 기법부터 딥러닝 시대까지 폭넓게 다룹니다.

* **전통적인 접근법**:
  * **비디오 오버세그멘테이션**: 비디오를 공간-시간적으로 균일한 영역으로 분할하는 기법 (예: 계층적 비디오 분할 [7], 시간적 슈퍼픽셀 [22], 슈퍼복셀 [3]).
  * **배경 차감(Background Subtraction)**: 배경이 미리 알려져 있고 카메라가 고정되어 있거나 예측 가능한 움직임을 보일 때 전경 객체를 추출하는 기법 (예: [24], [25]).
  * **모션 분할(Motion Segmentation)**: 움직이는 객체를 찾는 기법으로, 배경 차감의 특정 경우로 볼 수 있으며, 움직임 분석 [29] 및 통계적 기법 [32]에 기반합니다. 특히, **궤적 분할(Trajectory Segmentation)** [4], [33]–[36]은 장기적인 움직임 패턴을 활용했습니다.
  * **객체 가설/제안(Object Hypotheses/Proposals)**: AVOS에서 객체 후보를 생성하고 객체 영역 선택 문제로 접근하는 방식 [5], [38]–[40].
  * **반자동/대화형 VOS**: 광학 흐름 [8], [42], [44]을 이용하거나 로토스코핑 [47], [48], 스크리블 [8], [49]–[52] 등 광범위한 사용자 안내를 활용하는 기법.
  * **전통적인 VSS**: SVM과 비디오 오버세그멘테이션 기술에 의존 [12], [55]–[58].

* **관련 연구 분야**:
  * **시각 추적(Visual Tracking)**: 객체 위치 추론에 중점을 두며, 객체/카메라 움직임, 외형 변화, 가려짐 등 비디오 분할과 공통적인 과제를 공유합니다.
  * **이미지 시맨틱 분할(Image Semantic Segmentation)**: 딥러닝 기반 VSS의 발전 기반이 되었으며, VSS는 시간적 연속성을 활용하여 정확도와 효율성을 높였습니다.
  * **비디오 객체 탐지(Video Object Detection)**: 객체 제안 생성, 시간 정보 통합, 프레임 간 객체 연결 등 비디오 분할과 유사한 핵심 기술 단계와 과제를 공유합니다.

## 🛠️ Methodology

본 설문 조사는 딥러닝 기반 비디오 분할의 최신 발전을 체계적으로 검토합니다.

1. **문제 정의 및 분류**:
    * 비디오 분할을 출력 공간 $Y$에 따라 크게 두 가지로 분류합니다:
        * **비디오 객체 분할(VOS)**: 범주를 알 수 없는 지배적인 객체를 전경/배경으로 이진 분할합니다.
        * **비디오 시맨틱 분할(VSS)**: 미리 정의된 시맨틱 범주(예: 자동차, 건물, 보행자) 내의 객체를 다중 클래스로 분할합니다.
    * 추론 모드에 따라 VOS를 세 가지로 나눕니다:
        * **자동 비디오 객체 분할(AVOS)**: 수동 초기화 없이 자동으로 수행됩니다 (무감독 또는 제로샷).
        * **반자동 비디오 객체 분할(SVOS)**: 첫 프레임의 마스크, 경계 상자, 스크리블 등 제한된 사용자 개입을 허용합니다 (반지도 또는 원샷). 언어 기반 VOS(LVOS)는 SVOS의 하위 분야로 분류됩니다.
        * **대화형 비디오 객체 분할(IVOS)**: 분석 과정 전반에 걸쳐 사용자 안내(주로 스크리블)를 통합합니다.
    * 학습 패러다임에 따라 모델을 세 가지로 나눕니다:
        * **지도 학습(Supervised Learning)**: 대량의 잘 라벨링된 데이터가 필요합니다.
        * **비지도 학습(Unsupervised Learning) / 자기지도 학습(Self-supervised Learning)**: 라벨 없이 비디오 데이터의 고유 속성(예: 프레임 간 일관성)에서 파생된 유사 라벨을 사용합니다.
        * **약지도 학습(Weakly-supervised Learning)**: 태그, 경계 상자, 스크리블과 같이 비교적 쉽게 주석을 달 수 있는 데이터를 사용합니다.

2. **딥러닝 기반 VOS 모델 검토**:
    * **AVOS**: 딥러닝 모듈 기반, 픽셀 인스턴스 임베딩 기반, 단기/장기 정보 인코딩을 통한 End-to-end 방식, 비/약지도 기반, 인스턴스 수준 AVOS 등으로 분류하여 대표적인 모델들을 소개합니다.
    * **SVOS**: 온라인 미세 조정 기반, 전파 기반, 매칭 기반, 상자 초기화 기반, 비/약지도 기반 등으로 분류하여 대표 모델들을 소개합니다.
    * **IVOS**: 주로 상호작용-전파 기반 방법론들을 다룹니다.
    * **LVOS**: 시각-언어 정보 융합 전략에 따라 동적 합성곱 기반, 캡슐 라우팅 기반, 어텐션 기반 등으로 분류합니다.

3. **딥러닝 기반 VSS 모델 검토**:
    * **VSS(인스턴스 불인식)**: 정확도 향상 노력(광학 흐름, 순환 네트워크), 속도 향상 노력(키프레임, 특징 재활용), 반/약지도 기반 등으로 분류합니다.
    * **VIS(비디오 인스턴스 분할)**: Track-detect, Clip-match, Propose-reduce, Segment-as-a-whole 패러다임으로 분류합니다.
    * **VPS(비디오 파놉틱 분할)**: 이미지 파놉틱 분할 모델을 비디오 도메인으로 확장한 접근법을 다룹니다.

4. **비디오 분할 데이터셋 검토**:
    * Youtube-Objects [73], FBMS$_{59}$ [36], DAVIS$_{16}$ [17], DAVIS$_{17}$ [81], YouTube-VOS [95] 등 20개의 VOS 및 VSS 데이터셋의 특징을 요약하고 설명합니다.

5. **성능 비교 및 분석**:
    * 각 분야에서 가장 널리 사용되는 데이터셋(DAVIS$_{16}$ for AVOS, DAVIS$_{17}$ for Instance-level AVOS, SVOS, IVOS, A2D Sentence for LVOS, Cityscapes for VSS, YouTube-VIS for VIS, Cityscapes-VPS for VPS)을 선정합니다.
    * **평가 지표**: 객체 수준 AVOS의 Jaccard($J$), 경계 정확도($F$), 시간 안정성($T$) 등 각 작업에 특화된 평가 지표를 설명합니다.
    * 제공된 표(Table 7-14)를 통해 검토된 방법론들의 정량적 성능(정확도 및 FPS)을 집계하고 비교 분석합니다.

## 📊 Results

본 설문 조사는 주요 비디오 분할 분야에서 최신 딥러닝 기반 모델들의 성능을 벤치마킹하여 다음과 같은 결과를 제시합니다.

* **객체 수준 AVOS (DAVIS$_{16}$ val)**: RTNet [107]이 Jaccard $J$ 85.6%로 최고의 성능을 보였습니다. 이는 2017년 SFL [68]과 같은 초기 딥러닝 기반 방법을 크게 능가하는 수치입니다.
* **인스턴스 수준 AVOS (DAVIS$_{17}$ val)**: UnOVOST [178]가 Jaccard $J$ 67.9%로 현재까지 가장 우수한 성능을 달성했습니다.
* **SVOS (DAVIS$_{17}$ val)**: EGMN [100], LCM [172], RMNet [173] 등 메모리 증강 아키텍처(STM [149] 기반)를 활용한 상위 솔루션들이 J&F 평균 82-83%의 높은 성능을 보였습니다.
* **IVOS (DAVIS$_{17}$ val)**: Cheng et al. [189]이 AUC 84.9%, J@60 85.4%로 가장 뛰어난 성능을 기록했습니다.
* **LVOS (A2D Sentence test)**: CST [203]가 평균 IoU 56.1%로 최신 모델의 명확한 개선 추세를 보여주며, 초기 모델 A2DS [196] 대비 향상된 성능을 보였습니다.
* **VSS (Cityscapes val)**: EFC [223]가 IoU$_{\text{class}}$ 83.5%로 현재까지 가장 좋은 성능을 나타냈습니다.
* **VIS (YouTube-VIS val)**: Transformer 기반 아키텍처인 VisTR [238]과 Propose-Reduce [244]가 각각 mAP 40.1%와 47.6%로 SOTA 성능을 크게 향상시켰습니다.
* **VPS (Cityscapes-VPS test)**: ViP-DeepLab [241]이 VPQ 62.5%로 최고 성능을 달성했습니다.

이러한 결과는 딥러닝 기술이 비디오 분할의 다양한 하위 분야에서 상당한 발전을 이끌어냈음을 명확히 보여줍니다.

## 🧠 Insights & Discussion

본 설문 조사를 통해 몇 가지 중요한 통찰과 함께 이 분야의 한계점이 드러났습니다:

* **재현성 문제**: 많은 연구가 실험 설정에 대한 상세한 설명이나 소스 코드를 제공하지 않으며, 일부는 분할 마스크조차 공개하지 않습니다. 또한, 다양한 데이터셋과 백본 모델을 사용하여 공정한 비교가 어렵고 재현성을 저해합니다.
* **실행 시간 및 메모리 사용량 정보 부족**: 특히 AVOS, LVOS, VPS 분야에서 실행 시간과 메모리 사용량에 대한 보고가 부족합니다. 이는 많은 연구가 정확도에만 집중하고 효율성이나 메모리 요구 사항을 고려하지 않기 때문입니다. 하지만 모바일 기기나 자율 주행 자동차와 같이 컴퓨팅 자원이 제한된 실제 응용 시나리오에서는 이러한 정보가 매우 중요합니다.
* **일부 데이터셋의 성능 포화**: DAVIS$_{16}$ (AVOS), DAVIS$_{17}$ (SVOS), A2D Sentence (LVOS)와 같이 일부 광범위하게 연구된 비디오 분할 데이터셋에서는 성능이 거의 포화 상태에 이르렀습니다. 새로 제안된 데이터셋들이 더 큰 도전 과제를 제시하지만, 어떤 특정 도전 과제가 해결되었고 해결되지 않았는지 명확히 밝히지 않는 경향이 있습니다.

이러한 한계점을 바탕으로, 다음과 같은 향후 연구 방향이 제시됩니다:

* **장기 비디오 분할**: 비디오 편집과 같은 실제 응용 프로그램에 더 가깝게, 몇 분 단위의 장기 비디오 시퀀스에서 VOS 모델의 성능을 평가하고, 재탐지(re-detection) 능력을 향상시키는 연구가 필요합니다.
* **개방형 비디오 분할(Open World Video Segmentation)**: 기존 VSS 알고리즘은 닫힌 세계(closed-world) 패러다임에서 개발되어 알려지지 않은 범주에 대한 적응력이 부족합니다. 로봇공학, 자율 주행 등 실제 환경에서 미지의 범주를 식별할 수 있는 더 스마트한 VSS 시스템이 요구됩니다.
* **비디오 분할 하위 분야 간 협력**: VOS와 VSS는 가려짐, 변형, 빠른 움직임 등 많은 공통된 과제를 공유합니다. 이러한 작업들을 통합된 프레임워크 내에서 모델링하는 전례가 없으므로, 하위 분야 간의 긴밀한 협력이 필요합니다.
* **주석 효율적인 비디오 분할 솔루션**: 현재 최고 성능의 알고리즘은 대량의 주석 데이터가 필요한 완전 지도 학습(fully-supervised learning)에 기반합니다. 비디오 데이터의 높은 시간적 상관관계를 활용하여 비/약지도 학습을 통해 주석 부담을 줄이는 연구가 매력적인 방향입니다.
* **적응형 계산(Adaptive Computation)**: 비디오 프레임 간의 높은 상관관계를 활용하여 계산 비용을 줄이는 유연한 분할 모델 설계가 필요합니다. 네트워크 아키텍처가 입력에 따라 동적으로 변경되는 방식(예: 일부 네트워크만 활성화)을 탐색해야 합니다.
* **신경망 아키텍처 탐색(Neural Architecture Search, NAS)**: 현재 비디오 분할 모델은 수동으로 설계된 아키텍처에 기반하므로 최적화되지 않을 수 있습니다. NAS 기술을 사용하여 비디오 분할 네트워크 설계를 자동화하는 것이 유망한 방향입니다.

## 📌 TL;DR

이 논문은 딥러닝 기반 비디오 분할 분야의 포괄적인 설문조사로, 기존 연구들의 한계(오래되었거나 특정 분야에만 집중)를 해결합니다. 논문은 비디오 객체 분할(VOS)과 비디오 시맨틱 분할(VSS)을 주요 두 축으로 삼아, 각 분야의 하위 태스크(예: AVOS, SVOS, IVOS, LVOS, VIS, VPS)와 관련 방법론, 데이터셋을 상세히 분류하고 검토합니다. 또한, 주요 데이터셋에서 최신 모델들의 성능을 벤치마킹하여 결과를 비교합니다. 결론적으로, 이 분야는 딥러닝 덕분에 상당한 발전을 이루었지만, 재현성 부족, 효율성 지표의 부재, 일부 데이터셋의 성능 포화와 같은 중요한 과제들이 남아있음을 지적하고, 장기 비디오 분할, 개방형 세계 분할, 주석 효율적인 솔루션, 적응형 계산, 신경망 아키텍처 탐색 등 향후 연구 방향을 제안합니다.
