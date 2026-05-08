# Adapting SAM for Surgical Instrument Tracking and Segmentation in Endoscopic Submucosal Dissection Videos

Jieming Yu, Long Bai, Guankun Wang, An Wang, Xiaoxiao Yang, Huxin Gao, Hongliang Ren (2024)

## 🧩 Problem to Solve

본 논문은 내시경 점막하 박리술(Endoscopic Submucosal Dissection, ESD) 비디오에서 수술 도구를 정밀하게 추적하고 분할(Segmentation)하는 문제를 해결하고자 한다. 수술 도구의 정확한 위치와 움직임을 파악하는 것은 수술의 효율성을 높이는 데 매우 중요하지만, 다음과 같은 세 가지 핵심적인 난제가 존재한다.

첫째, 수술 도구 분할을 위한 대규모 데이터셋의 수동 어노테이션(Manual Annotation) 작업에 막대한 시간과 노동력이 소모된다. 둘째, 자연 이미지 분할에서 탁월한 성능을 보이는 Segment Anything Model (SAM)을 수술 도메인에 직접 적용할 경우, 사전 학습 데이터와 수술 영상 간의 큰 도메인 차이(Domain Gap)로 인해 성능이 저하된다. 셋째, SAM은 단일 이미지 기반 모델이므로 비디오 시퀀스에 적용했을 때 프레임 간의 시간적 일관성(Temporal Correspondence)을 유지하기 어려워 결과가 불안정하다는 한계가 있다.

따라서 본 연구의 목표는 수동 어노테이션의 필요성을 최소화하면서, 수술 영상 전체에서 도구의 분할 및 추적 정확도를 높이는 효율적인 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **LoRA(Low-Rank Adaptation)를 통한 SAM의 도메인 적응**과 **XMem++를 이용한 시공간적 추적의 결합**이다.

단순히 모든 파라미터를 미세 조정(Full Fine-tuning)하는 대신, 가벼운 LoRA 레이어를 추가하여 수술 도메인에 특화된 SAM 모델을 생성하고, 이렇게 얻은 정밀한 초기 마스크를 XMem++라는 최신 비디오 객체 분할(VOS) 모델에 전달함으로써 비디오 전체의 도구 추적을 자동화하는 2단계 파이프라인을 제안한다. 또한, 연구팀은 내시경 점막하 박리술(ESD) 수술 영상으로 구성된 새로운 데이터셋을 공개하여 벤치마크를 제공하였다.

## 📎 Related Works

논문에서는 기존의 SAM이 제로샷(Zero-shot) 일반화 능력이 뛰어나지만, 의료 영상 분야에서는 사전 학습 데이터의 부족으로 인해 한계가 있음을 지적한다. 특히 SAM을 비디오에 적용하려는 시도(예: Track Anything)가 있었으나, 시간적 일관성 유지 문제가 여전히 해결되지 않았음을 언급한다.

기존 방식과의 차별점은 다음과 같다. 첫째, SAM의 거대한 파라미터 수(ViT-B 기준 91M)로 인해 불가능에 가까운 전체 미세 조정 대신 LoRA를 사용하여 효율적인 적응을 꾀했다. 둘째, 매 프레임마다 프롬프트를 입력해야 하는 SAM의 상호작용적 특성을 극복하기 위해, 초기 프레임만 SAM으로 분할하고 이후는 XMem++의 메모리 기반 추적 방식을 사용하여 자동화를 달성했다.

## 🛠️ Methodology

전체 시스템 구조는 '학습 단계'와 '추론 단계'의 두 과정으로 구성된다.

### 1. SAM의 도메인 적응 (Fine-tuning with LoRA)

SAM 모델의 거대한 크기를 고려하여, 효율적인 학습을 위해 LoRA 방식을 채택한다.

- **구조적 변경**: SAM의 이미지 인코더(Image Encoder) 내의 각 ViT(Vision Transformer) 블록에 LoRA 레이어를 추가한다.
- **학습 전략**: 프롬프트 인코더(Prompt Encoder)와 이미지 인코더의 기존 가중치는 동결(Freeze)시키고, 추가된 LoRA 레이어만 학습시킨다.
- **하이퍼파라미터**: LoRA 행렬의 랭크(Rank)는 $512$로 설정하였으며, 총 $10$ 에포크(Epoch) 동안 EndoVis17 데이터셋을 사용하여 학습하였다.
- **입력 프롬프트**: 수술 도구의 위치를 지정하는 Bounding Box 프롬프트를 사용하며, 이는 이미지 임베딩과 융합되어 최종 마스크 디코더(Mask Decoder)를 통해 분할 결과물을 생성한다.

### 2. 비디오 추적 (Tracking with XMem++)

학습된 SAM 모델이 단일 이미지에 강점이 있다는 점을 활용하여, 비디오 전체를 처리하는 파이프라인을 구성한다.

- **절차**: 비디오의 초기 몇 프레임에 대해 Fine-tuned SAM을 적용하여 정밀한 분할 마스크를 생성한다.
- **추적 메커니즘**: 생성된 마스크를 XMem++ 모델의 입력으로 전달한다. XMem++는 영구 메모리 모듈(Permanent Memory Module)을 통해 이전 프레임의 객체 정보를 기억하며, 이를 바탕으로 나머지 비디오 프레임에서 도구를 추적한다.
- **효과**: 이 방식을 통해 XMem++의 기존 방식에서 필요했던 수동 프레임 어노테이션 과정을 SAM이 대체함으로써 전체 프로세스의 자동화를 구현한다.

## 📊 Results

### 실험 설정

- **데이터셋**: EndoVis17(학습 및 검증), EndoVis18(OOD 검증), ESD(자체 제작 신규 데이터셋, 3개 영상/300프레임)
- **비교 대상**: Original SAM, Track Anything
- **평가 지표**: $\text{mIoU}$ (mean Intersection over Union), $\text{mAcc}$ (mean Accuracy), $\text{mDice}$ (mean Dice coefficient)

### 정량적 결과

- **SAM 미세 조정 효과 (Table I)**:
  - EndoVis17에서 Original SAM 대비 $\text{mIoU}$가 $40.29\%$에서 $91.38\%$로 $51.09\%$p 상승하는 비약적인 성능 향상을 보였다.
  - EndoVis18(OOD 데이터셋)에서도 $\text{mIoU}$가 $32.99\%$에서 $85.28\%$로 크게 향상되어, 모델의 강건성(Robustness)이 입증되었다.
  - ESD 데이터셋에서도 $\text{mIoU}$가 $79.67\% \rightarrow 82.56\%$로 향상되었다.

- **비디오 추적 성능 (Table II)**:
  - ESD 데이터셋에 대해 'Fine-tuned SAM + XMem++' 조합은 Track Anything 대비 $\text{mIoU}$ $88.17\%$ (vs $86.75\%$), $\text{mDice}$ $96.16\%$ (vs $95.58\%$)로 더 우수한 성능을 기록하였다.

## 🧠 Insights & Discussion

본 연구는 파운데이션 모델(Foundation Model)인 SAM을 의료라는 특수 도메인에 효율적으로 이식하는 방법론을 제시하였다. 특히 LoRA를 통해 적은 비용으로 도메인 적응을 수행하고, 이를 VOS 모델인 XMem++와 결합하여 '정밀한 초기화 $\rightarrow$ 일관된 추적'이라는 효율적인 워크플로우를 구축한 점이 강점이다.

다만, 몇 가지 한계점과 논의 사항이 존재한다. 첫째, 초기 프레임에서 Bounding Box 프롬프트가 필요하므로 완전한 무감독(Unsupervised) 방식은 아니라는 점이다. 둘째, 사용된 ESD 데이터셋의 규모가 3개의 영상으로 매우 작아, 더 다양한 수술 케이스에 대한 일반화 성능 검증이 추가적으로 필요하다. 셋째, 실시간성(Real-time performance)에 대한 구체적인 분석이 부족하여 실제 수술 로봇 시스템에 탑재했을 때의 지연 시간(Latency) 문제가 해결되었는지는 명시되지 않았다.

## 📌 TL;DR

본 논문은 SAM을 수술 도구 분할에 최적화하기 위해 LoRA로 미세 조정하고, 이를 XMem++ 추적 알고리즘과 결합한 프레임워크를 제안한다. 이를 통해 수동 어노테이션 부담을 줄이면서도 수술 영상 내 도구 추적 및 분할 성능을 획기적으로 높였으며, 특히 신규 ESD 데이터셋에서 기존 Track Anything보다 우수한 성능을 입증하였다. 이 연구는 향후 수술 로봇의 지능형 인지 시스템 구축에 중요한 기초 기술이 될 가능성이 높다.
