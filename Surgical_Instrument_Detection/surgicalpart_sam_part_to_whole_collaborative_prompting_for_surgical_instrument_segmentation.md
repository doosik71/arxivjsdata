# SurgicalPart-SAM: Part-to-Whole Collaborative Prompting for Surgical Instrument Segmentation

Wenxi Yue, Jing Zhang, Kun Hu, Qiuxia Wu, Zongyuan Ge, Yong Xia, Jiebo Luo, and Zhiyong Wang (2024)

## 🧩 Problem to Solve

본 논문은 수술 도구 분할(Surgical Instrument Segmentation, SIS) 작업에서 발생하는 기존 방법론들의 한계를 해결하고자 한다. 수술 도구 분할은 수술 계획, 로봇 내비게이션, 기술 평가 등 다양한 하위 애플리케이션의 기초가 되는 중요한 작업이다.

현재의 SIS 접근 방식은 크게 두 가지 문제점을 가지고 있다. 첫째, 기존의 전문가 모델(Specialist Models)들은 방대한 양의 파라미터를 학습시켜야 하므로 개발 비용이 매우 높으며, 수술 현장에서 요구되는 인간-컴퓨터 상호작용 능력이 부족하다. 둘째, 최근 각광받는 Segment Anything Model (SAM)을 적용하려는 시도들이 있었으나, 일반적인 객체와 수술 도구 사이의 도메인 격차(Domain Disparity)로 인해 Zero-shot 성능이 낮으며, 수술 중 매 프레임마다 포인트나 박스 프롬프트를 제공하는 것은 현실적으로 불가능하다.

특히, 기존의 SAM 튜닝 방식(예: SurgicalSAM)은 도구를 하나의 단일 개체로 처리하여 도구의 복잡한 구조와 세부 디테일을 무시하며, 카테고리 기반의 프롬프트는 도구의 구조적 특성을 충분히 설명하기에 유연성과 정보량이 부족하다는 한계가 있다. 따라서 본 논문은 전문가의 도구 구조 지식을 통합하여 텍스트 프롬프트 기반으로 정밀한 수술 도구 분할을 수행하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 수술 도구가 여러 부분(Part)들의 조합으로 이루어져 있다는 전문가 지식을 SAM의 학습 과정에 명시적으로 통합하는 것이다. 이를 위해 다음과 같은 설계를 제안한다.

1. **Collaborative Prompts**: 단순히 도구의 카테고리 이름만 사용하는 것이 아니라, `{[part name] of [instrument category name]}` 형태의 텍스트 세트를 사용하여 카테고리 수준과 부분 수준의 정보를 동시에 제공한다.
2. **Cross-Modal Prompt Encoder**: 텍스트 프롬프트와 이미지 임베딩을 결합하여 도구의 각 부분에 특화된 변별력 있는 표현(Part-level representations)을 학습한다.
3. **Part-to-Whole Adaptive Fusion & Hierarchical Decoding**: 학습된 부분별 표현들을 적응적으로 융합하여 전체 도구의 표현을 생성하고, 전체와 부분 모두를 동시에 디코딩함으로써 구조적 이해도와 세부 디테일 캡처 능력을 동시에 향상시킨다.

## 📎 Related Works

### 1. Surgical Instrument Segmentation

초기 연구들은 U-Net 기반의 TernausNet과 같은 시맨틱 분할이나 Mask R-CNN 기반의 ISINet과 같은 인스턴스 분할 모델을 개발하였다. 최근에는 Mask2Former나 Transformer 기반의 모델들이 제안되었으나, 이러한 전문가 모델들은 모든 파라미터를 학습시켜야 하므로 비용이 높다.

### 2. Text Promptable Segmentation

자연어 프롬프트를 사용하여 유연성과 일반화 성능을 높이려는 시도가 있었으며, 최근에는 CLIP과 같은 거대 시각-언어 모델(VLM)을 활용한 TP-SIS 등이 제안되었다. 하지만 TP-SIS는 도구 부분 마스크를 단순한 감독 신호로만 사용하여 구조적 의존성을 무시하며, CLIP 이미지 인코더 전체를 미세 조정해야 하므로 학습 비용이 크다.

### 3. Segment Anything Model (SAM)

SAM은 강력한 일반화 능력을 갖추었으나, 의료 영상 도메인에서의 Zero-shot 성능은 낮다. 이를 해결하기 위해 도메인 특화 데이터를 통한 파인튜닝 연구가 진행되었으나, 대부분 상호작용성이 낮거나 수동 프롬프트(포인트, 박스)에 의존하며, 카테고리 ID 기반의 유연하지 못한 프롬프트를 사용한다는 한계가 있다.

## 🛠️ Methodology

SP-SAM은 입력 이미지 $I$와 특정 카테고리 $c$에 대한 Collaborative Prompts $T^{(c)}$를 받아 도구의 이진 마스크 $M^{(c)}$를 예측한다.

### 1. 전체 파이프라인 및 시스템 구조

전체 구조는 **SAM Image Encoder $\rightarrow$ Cross-Modal Prompt Encoder $\rightarrow$ Part-to-Whole Adaptive Fusion $\rightarrow$ SAM Decoder** 순으로 구성된다. 이때 SAM Image Encoder, CLIP Text Encoder, SAM Decoder의 출력 MLP는 고정(Frozen)되며, 나머지 가중치만 효율적으로 튜닝한다.

### 2. 주요 구성 요소

#### (1) Cross-Modal Prompt Encoder

이 모듈은 Collaborative Prompts를 이미지 임베딩과 상호작용시켜 부분별 Sparse 및 Dense 임베딩을 생성한다.

- **특징 추출**: `{[part p] of [category c]}` 형태의 텍스트를 CLIP Text Encoder로 인코딩하고, 이를 Transfer MLP를 통해 SAM의 임베딩 공간으로 전이시켜 $T^{part} \in \mathbb{R}^{P \times d}$를 얻는다.
- **교차 모달 인코딩**: 이미지 임베딩 $F^I$와 $T^{part}$ 사이의 유사도 맵 $S = T^{part} \times {F^I}^\top$를 계산한다. 이 $S$를 공간적 어텐션으로 사용하여 이미지 임베딩을 활성화한 $F'^I = S \circ F^I + F^I$를 생성하며, 이를 통해 각 부분의 Sparse 임베딩 $F^{part}_S$와 Dense 임베딩 $F^{part}_D$를 각각 MLP와 CNN으로 도출한다.

#### (2) Part-to-Whole Adaptive Fusion

부분별 임베딩을 전체 도구 임베딩 $\{F^S, F^D\}$로 통합하는 과정이다. 두 가지 어텐션 메커니즘을 통해 적응적 융합을 수행한다.

- **Category Part Attention**: 전문가 지식 행렬 $D^{CP}$ (카테고리 $c$에 부분 $p$가 존재하는지 여부)를 기반으로 카테고리별 가중치 $D^{c*}$를 적용한다.
- **Image Part Attention**: 이미지의 전역 기술자(Global Descriptor) $F^G$를 Global CNN으로 학습하고, 이를 $T^{part}$와 비교하여 이미지 내 실제 가시성 및 폐색(Occlusion) 상태를 반영한 가중치 $W = F^G \times {T^{part}}^\top$를 계산한다.

최종 융합 식은 다음과 같다.
$$F^S = F^{part}_S \circ \text{ReLU}(D^{c*})$$
$$F'_D = F^{part}_D \circ D^{c*} \circ W$$
$$F^D = \sum_{p=1}^P F'_D$$

#### (3) Hierarchical Decoding 및 학습 목표

SAM Decoder를 통해 전체 마스크와 부분 마스크를 동시에 예측하는 계층적 디코딩을 수행한다. 손실 함수는 전체 마스크와 각 부분 마스크에 대한 Dice Loss $L^D$의 합으로 정의된다.
$$L = L^D(M^{(c)}, G^{(c)}) + \sum_{p=1}^P d_{cp} L^D(M^{(c)}_p, G^{(c)}_p)$$
여기서 $d_{cp}$는 해당 카테고리에 부분이 포함되어 있는지 나타내는 지표이다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: EndoVis2017 및 EndoVis2018.
- **평가 지표**: Challenge IoU, IoU, mean class IoU (mc IoU).
- **비교 대상**: 전문가 모델(TernausNet, ISINet, TP-SIS 등) 및 SAM 기반 모델(SurgicalSAM, PerSAM 등).

### 2. 주요 결과

- **정량적 결과**: SP-SAM은 두 데이터셋 모두에서 SOTA(State-of-the-art) 성능을 달성하였다. 특히 EndoVis2018의 Challenge IoU에서 SurgicalSAM 대비 약 3.91%p, EndoVis2017에서 4.00%p 향상되었다.
- **효율성**: TP-SIS가 131.08M의 튜닝 파라미터를 사용하는 반면, SP-SAM은 단 8.62M의 파라미터만 튜닝하면서도 더 우수한 성능을 보였다.
- **정성적 결과**: 시각적 분석 결과, SP-SAM은 도구의 끝단(Tip)이나 경계선(Edge)과 같은 세밀한 디테일을 훨씬 더 정확하게 포착하였다. 이는 기존 방식들이 도구의 끝부분을 놓치거나 경계가 거칠게 표현되는 것과 대조적이다.

### 3. 절제 연구 (Ablation Study)

- Collaborative Prompts, Adaptive Fusion, Hierarchical Decoding의 모든 구성 요소가 통합되었을 때 성능이 극대화됨을 확인하였다.
- 특히 Adaptive Fusion 없이 부분 마스크를 단순 보조 신호로 사용했을 때보다, 제안된 융합 메커니즘을 사용했을 때 성능 향상 폭이 훨씬 컸다.
- Dense 임베딩이 Sparse 임베딩보다 성능에 더 큰 영향을 미치며, 두 임베딩을 모두 사용할 때 최적의 결과가 도출되었다.

## 🧠 Insights & Discussion

본 논문은 수술 도구와 같이 구조가 정형화되어 있고 세부 디테일이 중요한 객체를 분할할 때, 단순한 카테고리 정보보다는 **부분-전체(Part-to-Whole)의 계층적 구조 지식**을 모델에 주입하는 것이 매우 효과적임을 입증하였다.

특히 **Image Part Attention**을 통해 이미지 내에서의 실제 가시성을 고려함으로써, 수술 중 빈번하게 발생하는 도구의 폐색(Occlusion) 문제를 효과적으로 처리할 수 있었다. 또한, 매우 적은 수의 파라미터만 튜닝함으로써 거대 모델인 SAM을 특정 도메인에 효율적으로 적응시킬 수 있음을 보여주었다.

다만, 본 연구는 정적인 이미지 프레임에 집중하고 있어 수술 영상의 시간적 연속성(Temporal cues)을 활용하지 못했다는 점과, 인간 조직과 같은 배경 타겟에 대한 분할은 다루지 않았다는 한계가 있다.

## 📌 TL;DR

SP-SAM은 수술 도구의 구성 부분에 대한 전문가 지식을 텍스트 형태의 **Collaborative Prompts**로 구현하고, 이를 **적응적 융합(Adaptive Fusion)** 메커니즘을 통해 SAM에 통합한 효율적 튜닝 프레임워크이다. 이 연구는 최소한의 파라미터 튜닝만으로도 기존 전문가 모델 및 SAM 기반 모델들을 능가하는 정밀한 분할 성능을 달성하였으며, 향후 수술 로봇의 정밀 제어 및 자동화된 수술 분석 시스템의 기초 기술로 활용될 가능성이 매우 높다.
