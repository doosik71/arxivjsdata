# PaLM-E: An Embodied Multimodal Language Model

Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, Pete Florence

## 🧩 Problem to Solve

거대 언어 모델(LLM)은 복잡한 추론 능력을 보여주지만, 로봇 공학과 같은 실제 세계 문제에 적용될 때 "접지(grounding)" 문제에 직면합니다. LLM은 텍스트 데이터에만 의존하여 물리적 세계에 대한 이해가 부족하며, 실제 시각 및 물리적 센서 양식과 연결되지 않으면 장면의 기하학적 구성이 중요한 많은 로봇 작업에 충분하지 않습니다. 기존 시각-언어 모델(VLM)도 로봇 추론 작업을 직접 해결하지 못합니다.

## ✨ Key Contributions

- **일반주의적 임베디드 의사결정 에이전트 제안:** 다중 양식 LLM 훈련에 임베디드 데이터를 혼합하여 일반주의적이고 전이 학습된 다중 구현체(multi-embodiment) 의사결정 에이전트 훈련이 가능함을 입증했습니다.
- **효율적인 임베디드 추론기 개발:** 현재 최첨단 범용 VLM이 임베디드 추론 문제를 잘 해결하지 못하지만, 효율적인 임베디드 추론기 역할을 할 수 있는 유능한 범용 VLM을 훈련할 수 있음을 보여주었습니다.
- **새로운 아키텍처 아이디어 도입:** 신경 장면 표현(neural scene representations) 및 엔티티 레이블링 다중 양식 토큰(entity-labeling multimodal tokens)과 같은 새로운 아키텍처 아이디어를 제시했습니다.
- **정량적으로 유능한 시각 및 언어 일반주의자 입증:** PaLM-E가 임베디드 추론기일 뿐만 아니라 정량적으로 유능한 시각 및 언어 일반주의자임을 입증했습니다 (OK-VQA 벤치마크에서 SOTA 달성).
- **재앙적 망각(Catastrophic Forgetting) 감소:** 언어 모델 크기를 확장하면 다중 양식 미세 조정 시 재앙적 망각이 현저히 줄어든다는 것을 시연했습니다.

## 📎 Related Works

- **General Vision-Language Modeling:** 대규모 언어 및 비전 모델의 성공을 기반으로 하는 VLM (예: ViLBERT, Flamingo)이 이미지와 텍스트를 동시에 이해하여 시각 질의 응답(VQA), 캡셔닝, 객체 감지 등의 작업을 수행합니다. PaLM-E는 `multimodal sentences` 개념을 통해 여러 이미지를 유연하게 처리하며, Frozen(Tsimpoukelli et al., 2021)과 유사하게 LLM을 고정하고 비전 인코더를 최적화하지만, 더 넓은 범위의 입력 양식을 도입하고 성능을 능가합니다.
- **Actions-Output Models:** 비전 및 언어 입력을 임베디드 환경에서 직접 행동 예측(예: VIMA, Gato)을 위해 결합하는 연구들이 있습니다. PaLM-E는 고수준의 텍스트 명령을 생성하여 모델이 자체 예측에 기반하고 매개변수에 내재된 세계 지식을 직접 활용할 수 있도록 합니다.
- **LLMs in Embodied Task Planning:** LLM을 활용하여 임베디드 도메인에서 자연어 목표를 이해하고 계획을 생성하는 연구들이 진행되고 있습니다 (예: SayCan, Inner Monologue). 이러한 접근 방식은 종종 보조 모델이나 프롬프팅에 의존하여 접지 문제를 해결하는 반면, PaLM-E는 보조 모델 없이 계획을 직접 생성하도록 훈련됩니다.

## 🛠️ Methodology

PaLM-E의 핵심 아이디어는 사전 훈련된 언어 모델(PaLM)의 언어 임베딩 공간에 이미지, 상태 추정치 또는 기타 센서 양식과 같은 연속적인 **임베디드 관측치(embodied observations)**를 주입하는 것입니다.

1. **다중 양식 문장(Multi-modal Sentences):**
   - 입력은 텍스트와 (다중) 연속 관측치로 구성됩니다.
   - 이러한 관측치에 해당하는 다중 양식 토큰은 텍스트와 섞여 `multimodal sentences`를 형성합니다. 예를 들어, `Q: What happened between <img1> and <img2>?`와 같습니다.
   - PaLM-E는 디코더 전용 LLM으로, 이러한 접두사(prefix) 또는 프롬프트가 주어지면 텍스트 완성을 자기회귀적으로(autoregressively) 생성합니다.
2. **연속 관측치 주입:**
   - 연속 관측치($O$)는 인코더($\phi: O \rightarrow X_q$)를 통해 언어 토큰의 임베딩 공간($X \subset R^k$)과 동일한 차원의 벡터 시퀀스로 매핑됩니다.
   - 이 벡터들은 일반 임베디드 텍스트 토큰과 섞여 LLM의 접두사를 형성합니다.
   - 수학적으로는 각 접두사 벡터 $x_i$는 다음과 같이 구성됩니다:
     $$x_i = \begin{cases} \gamma(w_i) & \text{if } i \text{ is text token, or} \\ \phi_j(O_j)_i & \text{if } i \text{ corresponds to observation } O_j. \end{cases}$$
     여기서 $\gamma$는 단어 토큰 임베딩 함수이고, $\phi_j(O_j)_i$는 $O_j$ 관측치를 인코딩한 벡터 시퀀스의 $i$-번째 요소입니다.
3. **인코더(Encoders) 및 장면 표현:**
   - **State estimation vectors:** 장면의 객체 상태를 나타내는 벡터($s \in R^S$)를 MLP($\phi_{state}$)를 통해 언어 임베딩 공간으로 매핑합니다.
   - **Vision Transformer (ViT):** 이미지를 여러 토큰 임베딩으로 매핑하며, ViT의 임베딩 차원($\tilde{k}$)이 언어 모델과 다를 경우 학습된 어파인 변환($\psi$)을 통해 조정합니다. (예: ViT-4B, ViT-22B)
   - **Object-centric representations:** ViT 표현을 객체 인스턴스 마스크($M_j$)를 사용하여 개별 객체로 분해하여 $x_{j, 1:m} = \phi_{ViT}(M_j \circ I)$와 같이 나타냅니다.
   - **Object Scene Representation Transformer (OSRT):** 지상 진실 세분화(ground-truth segmentation) 없이 3D-중심 신경 장면 표현을 학습하여 장면을 객체 슬롯($o_j = \bar{\phi}_{OSRT}(I_{1:v})_j \in R^{\bar{k}}$)으로 분해합니다. 각 슬롯은 MLP($\psi$)를 통해 언어 임베딩 공간으로 투영됩니다.
   - **Entity referrals:** OSRT와 같은 객체 중심 표현의 경우, 입력 프롬프트의 객체에 해당하는 다중 양식 토큰을 `Object 1 is <obj1>. ... Object j is <objj>.`와 같이 레이블링하여 PaLM-E가 생성된 출력 문장에서 `objj`와 같은 특수 토큰을 통해 객체를 참조할 수 있도록 합니다.
4. **로봇 제어 루프와의 통합:**
   - PaLM-E가 임베디드 계획 또는 제어 작업을 해결하는 경우, 저수준 명령을 조건화하는 텍스트를 생성합니다.
   - PaLM-E는 순차적 의사결정을 생성하고, 이러한 결정은 로봇의 저수준 정책을 통해 실행됩니다.
   - 새로운 관측치가 발생하면 PaLM-E는 필요에 따라 계획을 다시 세울 수 있습니다. 이 과정은 자기회귀적으로(autoregressive manner) 진행됩니다.
5. **훈련 전략:**
   - PaLM-E는 사전 훈련된 PaLM(8B, 62B, 540B) 버전에 기반하며, 여기에 인코더를 통해 연속 관측치를 주입합니다.
   - **모델 동결(Model Freezing):** LLM을 고정하고 입력 인코더만 훈련하는 방법을 탐색합니다. 이는 LLM의 추론 능력을 활용하면서 접지(grounding) 문제를 해결하는 방식입니다.
   - **다중 작업 동시 훈련(Co-training across tasks):** 다양한 데이터셋(로봇 조작, VQA, 이미지 캡셔닝, 일반 언어 작업)을 혼합하여 훈련함으로써 모델의 전이 학습 효과를 극대화합니다. 로봇 데이터는 전체 데이터 혼합의 약 8.9%만 차지합니다.

## 📊 Results

- **로봇 조작 작업(TAMP, Language-Table, Mobile Manipulation):**
  - 다양한 로봇 작업(TAMP, Language-Table, Mobile Manipulation)에서 PaLM-E는 기존의 SOTA VLM(PaLI)이나 LLM 접지 알고리즘(SayCan)을 능가하는 성능을 보였습니다.
  - 특히, 데이터 효율성이 높아 소수의 훈련 예제로도 높은 성공률을 달성했습니다 (예: Language-Table의 경우 작업당 10~80개, TAMP의 경우 320개).
  - OSRT와 같은 3D-인식 객체 표현은 적은 데이터로도 가장 효과적인 입력 인코딩을 제공했습니다.
  - 실제 로봇 환경에서도 장기적인 조작 작업을 성공적으로 수행하고 교란에 강건한 모습을 보였습니다.
  - 새로운 객체 조합이나 미학습 객체에 대한 1-샷 또는 제로-샷 일반화 능력을 시연했습니다.
  - 모바일 조작 환경에서는 `affordance prediction` 및 `failure detection`에서 SOTA를 달성했습니다.
- **일반 시각-언어 작업:**
  - 단일 PaLM-E-562B 모델은 OK-VQA 벤치마크에서 SOTA 성능을 달성했으며, VQA v2 및 COCO 캡셔닝에서도 경쟁력 있는 결과를 보였습니다. 이는 PaLM-E가 로봇 추론기일 뿐만 아니라 유능한 VLM 일반주의자임을 입증합니다.
  - 단일 이미지 프롬프트로만 훈련되었음에도 다중 양식 사고의 사슬(multimodal chain-of-thought) 추론, 다중 이미지 추론과 같은 `emergent capabilities`를 보여주었습니다.
- **일반 언어 작업:**
  - 모델 규모가 커질수록(예: PaLM-E-562B) 다중 양식 훈련 중 언어 능력의 `재앙적 망각(catastrophic forgetting)`이 현저히 줄어들어, 원래 PaLM 모델의 NLG 성능 손실이 3.9%에 불과했습니다. 반면, 가장 작은 모델(PaLM-E-12B)에서는 87.3%의 성능 저하가 있었습니다.

## 🧠 Insights & Discussion

- **일반주의 모델과 전이 학습:** PaLM-E는 다양한 작업과 데이터셋에 대한 동시 훈련을 통해 각 개별 작업에 대해 따로 훈련된 모델보다 훨씬 뛰어난 성능을 보였고, 이는 `transfer learning`의 중요성을 강조합니다. 특히 로봇 환경 간 및 로봇-일반 VLM 작업 간의 전이가 두드러지게 나타났습니다.
- **데이터 효율성:** PaLM-E의 전이 학습 능력은 로봇 데이터를 적게 사용해도 높은 성능을 달성할 수 있게 합니다. OSRT와 같은 기하학적 입력 표현도 데이터 효율성을 높이는 데 기여합니다.
- **언어 능력 유지:** 다중 양식 훈련 중 언어 능력을 유지하는 두 가지 경로를 제시했습니다. 첫째, LLM을 `동결(freezing)`하고 입력 인코더만 훈련하는 방법은 실현 가능하지만, 로봇 작업에서 성능이 저하되는 경우가 있었습니다. 둘째, 전체 모델을 `종단 간(end-to-end) 훈련`할 경우, 모델 규모를 늘리면 재앙적 망각이 현저히 줄어들어 원래의 언어 성능을 더 많이 유지할 수 있습니다.
- **한계:** LLM을 고정하는 접근 방식은 로봇 작업에서 때때로 어려움을 겪었으며, 대규모 시각 데이터의 이점을 기하학적 입력 표현과 결합하는 것은 향후 연구 기회로 남아있습니다.

## 📌 TL;DR

PaLM-E는 거대 언어 모델(LLM)이 실제 세계의 센서 데이터와 연결되는 "접지" 문제에 직면한다는 점을 해결하기 위해 제안된 **임베디드 다중 양식 언어 모델**입니다. 이 모델은 이미지, 상태 추정치 등의 연속적인 관측치를 사전 훈련된 LLM (PaLM)의 임베딩 공간에 직접 주입하여 **다중 양식 문장**을 처리합니다. 다양한 로봇 조작 작업(TAMP, Language-Table, Mobile Manipulation)과 일반 시각-언어(VQA, 캡셔닝) 및 언어 작업에 대한 **다중 작업 동시 훈련**을 통해, PaLM-E는 **전이 학습**을 통해 로봇 작업에서 **높은 데이터 효율성**과 뛰어난 성능을 달성했습니다. 특히, PaLM-E-562B는 OK-VQA에서 SOTA를 달성하고, 모델 규모가 커질수록 언어 능력의 **재앙적 망각**이 크게 줄어드는 것을 보여주며, 다중 양식 사고의 사슬 추론과 같은 **새로운 능력(emergent capabilities)**을 시연했습니다.
