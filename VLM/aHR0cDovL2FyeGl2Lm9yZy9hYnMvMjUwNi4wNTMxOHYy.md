# Does Your 3D Encoder Really Work? When Pretrain-SFT from 2D VLMs Meets 3D VLMs

Haoyuan Li, Yanpeng Zhou, Yufei Gao, Tao Tang, Jianhua Han, Yujie Yuan, Dave Zhenyu Chen, Jiawang Bian, Hang Xu, Xiaodan Liang (2025)

## 🧩 Problem to Solve

본 논문은 3D Vision-Language Models(VLMs), 특히 **3D scene-centric VLM**이 2D VLM이나 3D object-centric 접근 방식에 비해 성능이 낮은 원인을 분석한다. 3D scene-centric 모델은 구조적으로 2D VLM과 매우 유사함에도 불구하고, 실제로는 3D 공간 구조를 제대로 이해하고 활용하는지에 대한 의문이 제기되었다.

연구의 핵심 목표는 3D VLM이 제공된 3D Encoder를 통해 실제로 장면을 "보고" 추론하는지, 아니면 데이터셋에 존재하는 언어적 패턴이나 정답 분포와 같은 **Shortcut Learning**에 의존하고 있는지를 정량적으로 분석하고, 이를 해결하기 위한 방안을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 3D scene-centric VLM이 3D 기하학적 정보에 극도로 낮게 의존하고 있음을 밝혀낸 것이다. 주요 기여 사항은 다음과 같다.

1. **3D Encoder 의존성 분석**: 3D Encoder의 가중치가 없거나 입력 토큰이 제거된 상태에서도 모델이 유사한 성능을 내는 현상을 발견하여, 3D VLM의 3D 공간 활용 능력이 제한적임을 증명하였다.
2. **Shortcut Learning 식별**: 모델이 3D 시각 정보 대신 질문-답변 쌍의 언어적 힌트와 빈번하게 등장하는 정답 분포(Frequent answer distributions)에 과적합(Overfitting)되어 있음을 밝혔다.
3. **3D-RDQA 데이터셋 제안**: 3D 토큰을 의도적으로 왜곡(Poisoning)하여 텍스트 힌트만으로는 풀 수 없게 만든 **3D Relevance Discrimination QA (3D-RDQA)** 데이터셋을 구축하여, 모델이 강제적으로 3D 공간 정보를 활용하도록 유도하고 이를 평가하는 기준을 마련하였다.

## 📎 Related Works

논문에서는 최근의 3D VLM을 인코더 설계 방식에 따라 세 가지로 분류한다.

1. **3D Object-centric VLM**: 장면을 객체들의 집합으로 보고, 개별 객체의 특성과 관계를 모델링한다. (예: LEO, Chat-3D)
2. **2D Image-based VLM**: 3D 장면을 여러 장의 2D 이미지 시퀀스로 렌더링하여 2D VLM의 능력을 활용한다. (예: LLaVA-3D)
3. **3D Scene-centric VLM**: 장면 전체를 하나의 개체로 처리하며 3D Point Cloud 등을 직접 인코딩한다. (예: 3D-LLM, LL3DA)

기존 연구들은 Contrastive Learning을 통한 모달리티 정렬(Alignment)에 집중했으나, 3D scene-centric 방식은 대규모 정렬 데이터셋의 부족으로 인해 3D Encoder의 세만틱 정보가 부족하며, 결과적으로 2D VLM에서 성공적이었던 Pre-train $\rightarrow$ SFT(Supervised Fine-Tuning) 파이프라인이 3D에서는 동일하게 작동하지 않는 한계가 있다.

## 🛠️ Methodology

### 분석 파이프라인

연구팀은 LL3DA를 베이스라인 모델로 설정하고, Qwen2-1.5B LLM과 Q-Former 프로젝터를 사용하였다. 분석을 위해 다음과 같은 실험적 설계를 도입하였다.

1. **Encoder Ablation**: 3D Encoder의 사전 학습 가중치를 제거하거나, Encoder 출력을 0으로 설정하여 성능 변화를 관찰한다.
2. **ScanQA-Choice**: 기존의 주관식 ScanQA를 객관식 형태로 변환하여, 3D 입력 없이 텍스트만으로 정답을 맞출 수 있는 확률을 측정한다.
3. **Data Distribution Analysis**: 정답 분포의 불균형이 모델의 예측에 미치는 영향을 분석하기 위해 데이터 밸런싱 실험을 수행한다.

### 3D-RDQA 데이터셋 설계

Shortcut Learning을 타파하기 위해, 본 논문은 **Relevance Discrimination** 개념을 도입한다.

- **구조**: 하나의 질문에 대해 '정상 3D 토큰'을 가진 쌍과 '왜곡된(Poisoned) 3D 토큰'을 가진 쌍으로 구성된 데이터셋이다.
- **왜곡 방법**: 해당 질문과 상관없는 다른 장면(Scene)에서 추출한 3D 토큰을 주입하여, 텍스트 정보와 시각 정보 간의 불일치를 생성한다.
- **목표**: 모델이 텍스트 힌트에만 의존한다면 두 쌍을 구분하지 못하겠지만, 실제로 3D 장면을 이해한다면 왜곡된 토큰이 주어졌을 때 이를 식별해내야 한다.

### 3D-VG (Visual Grounding) 정규화 공식

논문은 3D-VG 작업에서 바운딩 박스 좌표를 정규화하기 위해 다음 수식을 사용한다.
$$x = \frac{x - x_{min}}{x_{max} - x_{min}} \times g, \quad y = \frac{y - y_{min}}{y_{max} - y_{min}} \times g, \quad z = \frac{z - z_{min}}{z_{max} - z_{min}} \times g$$
여기서 $g$는 정규화된 그리드의 최대값(255)이다. 또한, 박스의 크기($w, h, l$)에 대해 **Signed Normalization**과 **Min-zero Normalization** 두 가지 방식을 비교 분석하였다.

## 📊 Results

### 주요 분석 결과

- **Encoder 무용론**: Table 1에서 3D Encoder 가중치를 사용하지 않거나($\times$), Encoder 출력을 제거한 경우에도 BLUE-4, CIDEr 등의 지표가 크게 떨어지지 않았다. 이는 모델이 3D 토큰보다 Q-Former의 latent query에 의존하고 있음을 시사한다.
- **Pre-training의 낮은 효율**: 2D VLM과 달리 3D VLM에서는 Pre-train 단계가 SFT 성능 향상에 결정적인 기여를 하지 못했다 (Table 2).
- **데이터 스케일링의 한계**: 데이터 양을 늘렸을 때 초기에는 성능이 향상되나, 약 135k 샘플 이후부터는 성능 향상이 정체되는 현상이 관찰되었다 (Table 4).
- **텍스트 의존성**: ScanQA-Choice 실험 결과, 3D 입력이 전혀 없는 상태에서도 매우 높은 정확도를 기록하였다. 이는 모델이 3D 공간을 이해하는 것이 아니라 질문-답변 간의 통계적 관계를 암기했음을 의미한다.

### 3D-RDQA 검증 결과

3D-RDQA 데이터셋을 적용했을 때, 결과는 극명하게 달라졌다.

- **3D 입력이 없는 모델**: 정확도가 $0\%$로 나타났다. 이는 모델이 텍스트만으로는 정답을 찾을 수 없게 설계된 3D-RDQA의 특성 때문이다.
- **정상 모델**: Pre-train과 SFT를 거친 모델은 3D 토큰의 유효성을 판단하여 정답을 맞혔으며, 이 과정에서 3D Encoder와 Pre-training 단계가 성능 향상에 필수적임이 확인되었다 (Table 9).

## 🧠 Insights & Discussion

### 강점 및 발견

본 연구는 3D VLM 연구 커뮤니티가 간과했던 **"시각적 맹목성(Visual Blindness)"** 문제를 정면으로 다루었다. 단순히 벤치마크 점수를 올리는 것이 아니라, 모델이 실제로 어떤 정보를 사용하여 답을 내는지 분석함으로써 3D VLM의 평가 방식에 대한 경종을 울렸다.

### 한계 및 비판적 해석

1. **객관식 포맷의 한계**: 3D-RDQA가 객관식 형태로 설계되어 있어, 이를 실제 open-ended한 3D 추론 작업으로 확장했을 때 동일한 효과를 거둘 수 있을지는 추가 연구가 필요하다.
2. **데이터셋의 편향**: 3D 데이터셋 자체가 가진 반복적인 패턴(예: "it is to the..."와 같은 정형화된 표현)이 Shortcut Learning을 가속화하고 있으며, 이는 단순히 모델 아키텍처의 문제가 아니라 데이터 구축 단계의 근본적인 문제임을 시사한다.

### 결론적 논의

결국 3D scene-centric VLM의 성능 저하는 인코더의 성능 부족보다는, **학습 과정에서 모델이 굳이 어려운 3D 기하학 정보를 학습할 필요 없이 텍스트만으로 정답을 맞출 수 있는 "쉬운 길(Shortcut)"을 찾았기 때문**이다. 따라서 향후 연구는 단순한 데이터 증강보다는 3D-RDQA와 같이 시각적 정렬을 강제하는 정교한 학습 전략이 필요하다.

## 📌 TL;DR

현재의 3D scene-centric VLM들은 3D Encoder를 통해 장면을 이해하는 것이 아니라, 텍스트 질문의 패턴과 정답 분포를 암기하여 답하는 **Shortcut Learning**에 의존하고 있다. 이를 증명하기 위해 3D 입력 없이도 높은 성능을 내는 현상을 분석했으며, 이를 해결하기 위해 시각 정보와 텍스트 정보의 일치 여부를 판단하게 만드는 **3D-RDQA 데이터셋**을 제안하여 진정한 3D 공간 이해를 유도하고 평가할 수 있는 기반을 마련하였다.
