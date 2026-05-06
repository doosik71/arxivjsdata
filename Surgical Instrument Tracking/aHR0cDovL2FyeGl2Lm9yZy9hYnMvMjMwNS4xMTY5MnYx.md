# Surgical-VQLA: Transformer with Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery

Long Bai, Mobarakol Islam, Lalithkumar Seenivasan, and Hongliang Ren (2023)

## 🧩 Problem to Solve

본 논문은 로봇 수술 영상에서 주니어 레지던트들이 전문가의 도움 없이도 수술 장면과 활동을 이해할 수 있도록 돕는 수술 질의응답(Surgical Question-Answering) 시스템 구축을 목표로 한다.

기존의 Visual Question Answering (VQA) 방식들은 다음과 같은 세 가지 주요 문제점을 가지고 있다. 첫째, 수술 도메인에서는 데이터셋의 규모가 작고 Bounding Box 주석(annotation)이 부족하여 신뢰할 만한 객체 검출(Object Detection) 모델을 확보하기 어렵다. 둘째, 텍스트와 이미지라는 서로 다른 성격의 이종 모달리티(Heterogeneous Modalities)를 결합하는 방식이 단순 결합(naive fusion)에 그쳐 효율성이 떨어진다. 셋째, 복잡한 수술 시나리오에서 답변의 근거가 되는 특정 영역을 함께 제시하는 localized answering 기능이 결여되어 있다.

따라서 본 연구는 답변을 예측함과 동시에 질문과 관련된 특정 수술 영역을 함께 찾아내는 Visual Question Localized-Answering (VQLA) 모델을 제안하여, 사용자가 "무엇(what)"과 "어디(where)"를 알게 함으로써 최종적으로 "왜(why)" 그런 결과가 나왔는지를 추론할 수 있게 하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 객체 검출기(Object Detector)에 의존하지 않는 **Detection-free** 구조를 채택하고, 이종 모달리티 간의 효율적인 융합을 위해 **Gated Vision-Language Embedding (GVLE)** 모듈을 도입한 것이다.

또한, 답변의 정확도뿐만 아니라 위치 예측의 정밀도를 높이기 위해 **Generalized Intersection over Union (GIoU)** 손실 함수를 통합하여, 분류(Classification)와 국지화(Localization)를 동시에 수행하는 엔드-투-엔드(End-to-End) 프레임워크를 설계하였다.

## 📎 Related Works

기존의 VQA 모델들(예: VisualBERT, VisualBERT ResMLP)은 주로 객체 검출 모델을 통해 제안된 영역(Object Proposals)에서 시각적 특징을 추출하여 텍스트 임베딩과 결합하는 방식을 사용한다. 하지만 이러한 접근 방식은 다음과 같은 한계가 있다.

1. **의존성 문제**: VQA 모델의 성능이 객체 검출 모델의 성능에 종속되며, 검출 단계의 작은 오류가 VQA 결과에 큰 영향을 미친다.
2. **전역 문맥 상실**: 특정 객체 영역만 추출하고 배경 정보를 무시함으로써, VQA에 필수적인 전체 장면 이해(Global Scene Understanding) 능력이 제한된다.
3. **효율성 저하**: 여러 단계의 네트워크(검출 $\rightarrow$ 특징 추출 $\rightarrow$ VQA)를 거쳐야 하므로 계산 비용이 높고 실시간 적용이 어렵다.

본 논문은 이러한 한계를 극복하기 위해 객체 검출 단계를 생략하고 전체 이미지 특징을 직접 활용하는 구조를 제안함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 전체 파이프라인 구조

Surgical-VQLA 시스템은 **ResNet18** 기반의 시각적 특징 추출기, 사용자 정의 **Tokenizer**, **GVLE** 모듈, 그리고 **Vision Transformer (ViT)** 인코더로 구성된다. 최종적으로 ViT의 출력은 답변을 예측하는 Classification Head와 위치를 예측하는 Localization Head로 나뉘어 병렬적으로 처리된다.

### 주요 구성 요소 및 역할

1. **Gated Vision-Language Embedding (GVLE)**:
   텍스트와 이미지 특징의 서로 다른 중요도를 조절하기 위해 게이트 메커니즘을 사용한다. 각 모달리티의 특징은 $\tanh$ 활성화 함수를 통해 내부 표현으로 인코딩되며, 게이트 노드 $\omega$가 각 임베딩 정보의 유용성을 판단하여 가중치를 조절한다.

   결합 공식은 다음과 같다.
   $$\omega = \alpha(\theta_\omega \cdot [f \| e])$$
   $$\Upsilon = \omega * \tanh(\theta_f \cdot f) + (1 - \omega) * \tanh(\theta_e \cdot e)$$
   여기서 $f$와 $e$는 각각 시각적 및 단어 임베딩을 나타내며, $[ \cdot \| \cdot ]$는 연결(concatenation) 연산, $\theta_\omega, \theta_f, \theta_e$는 학습 가능한 파라미터이다. 최종 출력 $\Upsilon$는 ViT 인코더의 입력으로 들어간다.

2. **Prediction Heads**:
   - **Classification Head**: ViT의 출력을 선형 층과 Softmax를 통해 통과시켜 최종 답변 클래스를 예측한다.
   - **Localization Head**: 3층의 Perceptron과 ReLU 활성화 함수를 가진 Feed-Forward Network (FFN)를 사용하여 바운딩 박스의 정규화된 좌표(높이, 너비, 중심 좌표)를 예측한다.

### 학습 절차 및 손실 함수

모델은 분류 손실과 검출 손실을 함께 최적화하는 Joint Training 방식으로 학습된다.

- **분류 손실**: 단순 Cross-Entropy ($L_{CE}$) 손실을 사용한다.
- **검출 손실**: $L_1$ 손실과 **GIoU (Generalized Intersection over Union)** 손실을 결합하여 사용한다. GIoU 손실은 겹치는 영역뿐만 아니라 겹치지 않는 영역까지 고려하여 바운딩 박스 회귀의 정확도를 높인다.
  
  $$L_{GIoU} = 1 - \left( \frac{|b_g \cap b_p|}{|b_g \cup b_p|} - \frac{|B(b_g, b_p) \setminus (b_g \cup b_p)|}{|B(b_g, b_p)|} \right)$$
  여기서 $b_g$는 정답(Ground Truth) 박스, $b_p$는 예측 박스, $B(b_g, b_p)$는 두 박스를 모두 포함하는 최소 크기의 박스를 의미한다.

최종 손실 함수 $L$은 다음과 같이 정의된다.
$$L = L_{CE} + (L_{GIoU} + L_1)$$

## 📊 Results

### 실험 설정

- **데이터셋**: MICCAI 챌린지의 EndoVis-17 및 18 데이터를 활용하여 VQLA 전용 데이터셋(EndoVis-18-VQLA, EndoVis-17-VQLA)을 구축하였다.
- **비교 대상**: VisualBERT, VisualBERT ResMLP.
- **지표**: Accuracy (Acc), F-Score, mean IoU (mIoU), 처리 속도 (FPS).

### 주요 결과

1. **정량적 성능**: 제안된 GVLE-LViT 모델은 두 데이터셋 모두에서 기존 SOTA 모델들보다 높은 Acc, F-Score, mIoU를 기록하였다. 특히 전체 이미지 특징을 사용한 경우가 객체 검출 기반 특징을 사용한 경우보다 일관되게 우수한 성능을 보였다.
2. **실시간성**: 객체 검출 네트워크를 제거함으로써 처리 속도가 8배 이상 향상되어 **150.6 FPS**를 달성하였으며, 이는 실시간 수술 지원 시스템으로의 적용 가능성을 시사한다.
3. **Ablation Study**:
   - **손실 함수**: $L_{CE}$와 $L_1$만 사용했을 때보다 GIoU 손실을 추가했을 때 답변 예측과 위치 국지화 성능이 모두 유의미하게 향상되었다.
   - **융합 기법**: GVLE 방식이 기존의 ConCAT, AFF, iAFF 방식보다 우수한 융합 성능을 보였다.

## 🧠 Insights & Discussion

본 연구의 강점은 수술 도메인에서 부족한 바운딩 박스 주석 데이터 문제를 해결하기 위해 Detection-free 구조를 제안하고, GVLE를 통해 이종 모달리티의 효율적인 결합을 이루어낸 점이다. 특히, 단순한 VQA를 넘어 '위치 정보'를 함께 제공함으로써 모델의 답변에 대한 신뢰도를 사용자가 직접 판단할 수 있게 한 점이 인상적이다. 예를 들어, 모델이 예측한 위치가 실제 수술 도구나 조직과 동떨어져 있다면 사용자는 해당 답변이 부정확할 가능성이 높다고 판단할 수 있다.

다만, 본 논문에서는 답변의 신뢰도를 정량적으로 예측하는 기능까지는 구현하지 않았으며, 이를 미래 연구 과제로 남겨두었다. 또한, 더 복잡한 수술 시나리오나 까다로운 질의응답 쌍을 포함한 대규모 데이터셋에서의 검증이 추가로 필요할 것으로 보인다.

## 📌 TL;DR

이 논문은 로봇 수술 영상에서 질문에 대한 답변과 그 근거 영역을 동시에 예측하는 **Surgical-VQLA** 시스템을 제안한다. 객체 검출기 없이 전체 이미지 특징을 활용하는 **GVLE-LViT** 구조와 **GIoU 손실 함수**를 도입하여, 기존 모델보다 높은 정확도와 압도적인 처리 속도(150.6 FPS)를 달성하였다. 이 연구는 향후 수술 교육 및 실시간 수술 보조 시스템의 신뢰성을 높이는 핵심 기술로 활용될 가능성이 크다.
