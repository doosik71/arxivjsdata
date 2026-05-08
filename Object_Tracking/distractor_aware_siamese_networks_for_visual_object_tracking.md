# Distractor-aware Siamese Networks for Visual Object Tracking

Zheng Zhu, Qiang Wang, Bo Li, Wei Wu, Junjie Yan, and Weiming Hu (2018)

## 🧩 Problem to Solve

본 논문은 시각적 객체 추적(Visual Object Tracking) 분야에서 Siamese network 기반 추적기들이 겪고 있는 고질적인 문제인 'semantic distractor(의미론적 방해 요소)'에 의한 성능 저하 문제를 해결하고자 한다.

기존의 Siamese 추적기들은 주로 전경(foreground)과 의미 없는 배경(non-semantic background)을 구분하는 법을 학습한다. 그러나 실제 환경에서는 타겟과 외형이 유사한 다른 객체들이 배경에 존재하는 경우가 많으며, 이러한 semantic distractor들은 추적기가 타겟을 놓치고 엉뚱한 객체로 전이되는 'drifting' 현상을 유발한다. 또한, 대부분의 Siamese 추적기는 모델을 온라인으로 업데이트하지 않아 타겟의 외형 변화에 취약하며, 국소적인 탐색 영역(local search region)만을 사용하기 때문에 완전 가려짐(full occlusion)이나 시야 이탈(out-of-view)과 같은 상황에서 타겟을 재발견하지 못하는 한계가 있다.

따라서 본 논문의 목표는 오프라인 학습 단계에서 distractor를 인식할 수 있는 변별력 있는 특징(feature)을 학습하고, 추론 단계에서 이를 효과적으로 억제하며, 나아가 장기 추적(long-term tracking)이 가능하도록 시스템을 확장하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 학습과 추론 전 과정에 걸쳐 distractor를 명시적으로 고려하는 **DaSiamRPN(Distractor-aware Siamese Region Proposal Networks)** 프레임워크를 설계하는 것이다.

1. **Distractor-aware Training**: 학습 데이터의 불균형(배경 vs 의미론적 객체)이 변별력 저하의 원인임을 분석하고, 다양한 semantic negative pairs를 생성하여 학습시킴으로써 특징 추출기의 변별력을 극대화한다.
2. **Distractor-aware Incremental Learning**: 추론 과정에서 현재 비디오 도메인의 특성에 맞게 일반적인 임베딩을 전이시키고, 타겟 주변의 distractor 정보를 활용하여 타겟의 응답 값을 재정렬하는 증분 학습 모듈을 제안한다.
3. **Local-to-Global Search Strategy**: 장기 추적을 위해 추적 실패 여부를 감지하고, 탐색 영역을 국소 영역에서 전역 영역으로 점진적으로 확장하는 전략을 도입하여 시야 이탈 및 가려짐 문제를 해결한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 소개한다.

- **Siamese Networks based Tracking**: SINT, SiamFC, GOTURN 등은 사전 학습된 Siamese 유사도 함수를 고정된 방식으로 사용하여 빠른 속도를 구현했다. 최근의 SiamRPN은 Region Proposal Network(RPN)를 도입하여 추적을 일종의 one-shot local detection 문제로 정의하였다. 그러나 이러한 방식들은 대부분 온라인 업데이트가 불가능하여 외형 변화에 취약하다.
- **Features for Tracking**: 기존 연구들은 색상 히스토그램이나 사전 학습된 CNN 특징을 사용했다. 하지만 단순한 특징은 배경 clutter 상황에서 robustness가 떨어진다는 한계가 있다.
- **Long-term Tracking**: TLD, MUSTer 등은 단기 추적기와 검출기(detector)를 결합하여 장기 추적을 수행한다. 하지만 이러한 방식들은 대개 속도가 느리거나 딥러닝 기반의 효율적인 Siamese 구조를 충분히 활용하지 못했다.

DaSiamRPN은 기존 Siamese 추적기의 빠른 속도를 유지하면서도, 학습 단계의 데이터 구성 최적화와 온라인 distractor 억제 메커니즘을 통해 기존의 한계를 극복하고자 한다.

## 🛠️ Methodology

### 1. Distractor-aware Training (오프라인 학습)

특징 추출기의 변별력을 높이기 위해 다음과 같은 학습 전략을 사용한다.

- **Positive Pairs 확장**: 기존 비디오 데이터셋(VID, Youtube-BB)의 카테고리가 적은 문제를 해결하기 위해, ImageNet Detection 및 COCO Detection과 같은 대규모 정지 이미지 데이터셋을 활용한다. 이미지 증강(Translation, Resize, Grayscale 등)을 통해 정지 이미지로부터 학습용 쌍을 생성한다.
- **Semantic Negative Pairs 생성**: 단순히 배경을 negative로 쓰는 것이 아니라, 타겟과 같은 카테고리 내의 다른 객체(intraclass distractors)와 다른 카테고리의 객체들을 negative pair로 명시적으로 추가한다. 이는 모델이 foreground/background 구분 수준을 넘어, 인스턴스 레벨(instance-level)의 세밀한 차이를 학습하게 한다.
- **데이터 증강**: 일반적인 변형 외에도 시각적 추적에서 자주 발생하는 모션 블러(motion blur)를 명시적으로 추가하여 강건성을 높인다.

### 2. Distractor-aware Incremental Learning (온라인 추론)

추론 단계에서는 일반적인 특징 공간을 현재 비디오 도메인에 최적화하기 위해 distractor-aware 모듈을 사용한다.

기본적인 Siamese 유사도 측정 함수는 다음과 같다.
$$f(z, x) = \phi(z) \star \phi(x) + b \cdot \mathbf{1}$$
여기서 $z$는 템플릿, $x$는 후보 이미지, $\star$는 cross-correlation을 의미한다.

DaSiamRPN은 NMS를 통해 타겟 $z_t$ 외에 높은 점수를 가진 잠재적 distractor들의 집합 $D = \{d_1, d_2, \dots, d_n\}$을 수집한다. 이후, distractor의 영향력을 억제하기 위해 다음과 같은 재정렬(re-rank) 목적 함수를 사용한다.
$$q = \text{argmax}_{p_k \in P} \left( f(z, p_k) - \frac{\hat{\alpha}}{\sum_{i=1}^{n} \alpha_i} \sum_{i=1}^{n} \alpha_i f(d_i, p_k) \right)$$
여기서 $\hat{\alpha}$는 distractor 학습의 영향력을 조절하는 가중치이며, $\alpha_i$는 각 distractor의 가중치이다.

연산 효율성을 위해 cross-correlation의 선형 결합 성질을 이용하여 위 식을 다음과 같이 변형하여 계산한다.
$$q = \text{argmax}_{p_k \in P} \left( \phi(z) - \frac{\hat{\alpha} \sum_{i=1}^{n} \alpha_i \phi(d_i)}{\sum_{i=1}^{n} \alpha_i} \right) \star \phi(p_k)$$

나아가, 시간적 흐름에 따라 타겟 템플릿과 distractor 템플릿을 학습률 $\beta_t$를 통해 증분적으로 업데이트한다.
$$q_{T+1} = \text{argmax}_{p_k \in P} \left( \frac{\sum_{t=1}^{T} \beta_t \phi(z_t)}{\sum_{t=1}^{T} \beta_t} - \frac{\sum_{t=1}^{T} \beta_t \hat{\alpha} \sum_{i=1}^{n} \alpha_i \phi(d_{i,t})}{\sum_{t=1}^{T} \beta_t \sum_{i=1}^{n} \alpha_i} \right) \star \phi(p_k)$$

### 3. Long-term Tracking 확장

장기 추적을 위해 **Local-to-Global Search Region** 전략을 도입한다.

- **실패 감지**: distractor-aware 모듈을 통해 계산된 detection score가 추적의 품질을 잘 반영한다는 점을 이용한다. 점수가 특정 임계값 이하로 떨어지면 추적 실패로 간주한다.
- **탐색 영역 확장**: 추적 실패가 감지되면, 탐색 영역의 크기를 일정 단계(constant step)로 점진적으로 키우며 타겟을 재탐색한다. 타겟이 다시 발견되어 점수가 회복되면 다시 국소 탐색 모드로 전환한다. 이 과정에서 Bounding Box Regression을 사용하므로 시간 소모가 큰 이미지 피라미드 전략 없이도 효과적인 재탐색이 가능하다.

## 📊 Results

### 실험 설정

- **데이터셋**: VOT2015, VOT2016, VOT2017, OTB2015, UAV20L, UAV123.
- **지표**: Accuracy (A), Robustness (R), Expected Average Overlap (EAO), AUC, Precision.
- **환경**: NVIDIA TITAN X GPU 사용.

### 주요 결과

- **단기 추적 (VOT)**: VOT2016에서 EAO 0.411을 기록하며 기존 SOTA였던 ECO(0.375) 대비 상대적으로 **9.6%의 성능 향상**을 보였다. VOT2017에서도 EAO 0.326으로 1위를 차지했다.
- **장기 추적 (UAV20L)**: AUC 61.7%를 달성하여 기존 최우수 추적기 대비 상대적으로 **35.9% 향상**된 결과를 보였다. 특히 Full Occlusion과 Background Clutter 상황에서 SiamRPN 대비 각각 153.1%, 393.2%의 압도적인 성능 향상을 보였다.
- **속도**: 단기 추적 시 **160 FPS**, 장기 추적 시 **110 FPS**의 실시간 이상의 속도로 동작한다.
- **Ablation Study**: detection data 추가 $\rightarrow$ semantic negative pairs 추가 $\rightarrow$ distractor-aware updating 추가 $\rightarrow$ long-term module 추가 순으로 성능이 단계적으로 향상됨을 확인하였다. (VOT2016 EAO: 0.344 $\rightarrow$ 0.368 $\rightarrow$ 0.389 $\rightarrow$ 0.411)

## 🧠 Insights & Discussion

본 논문은 Siamese 네트워크의 성능 한계가 단순히 네트워크 구조의 문제가 아니라, **학습 데이터의 분포(distribution) 문제**에서 기인함을 날카롭게 지적하였다. 배경과 타겟을 구분하는 쉬운 샘플(easy negatives) 위주의 학습에서 벗어나, 의미론적으로 유사한 객체(hard negatives)를 명시적으로 학습시킨 것이 변별력 향상의 핵심이었다.

또한, 고정된 특징 추출기만을 사용하는 대신, 추론 과정에서 distractor 정보를 이용해 현재 도메인에 맞게 유사도 함수를 최적화하는 '가벼운 온라인 업데이트' 방식을 제안함으로써, 속도와 정확도라는 두 마리 토끼를 모두 잡았다.

다만, 본 논문에서 제시한 Local-to-Global 전략은 타겟이 완전히 시야에서 사라졌을 때 탐색 영역을 확장하는 단순한 방식이므로, 타겟이 매우 빠르게 이동하거나 예측 불가능한 방향으로 사라지는 경우에 대한 정교한 예측 메커니즘은 부족해 보인다. 그럼에도 불구하고, Siamese 구조 내에서 distractor를 효율적으로 처리하는 방법론을 제시했다는 점에서 큰 의의가 있다.

## 📌 TL;DR

본 논문은 Siamese 네트워크 기반 추적기가 타겟과 유사한 객체(distractor) 때문에 발생하는 drifting 문제를 해결하기 위해 **DaSiamRPN**을 제안한다. **(1) Semantic negative pair를 활용한 오프라인 학습**, **(2) 온라인 distractor 억제 및 증분 학습 모듈**, **(3) 장기 추적을 위한 Local-to-Global 탐색 전략**을 통해 정확도와 속도를 동시에 확보하였다. 실험 결과 VOT 및 UAV 벤치마크에서 SOTA 성능을 달성하였으며, 110~160 FPS의 매우 빠른 속도로 동작하여 실제 실시간 시스템 적용 가능성이 매우 높다.
