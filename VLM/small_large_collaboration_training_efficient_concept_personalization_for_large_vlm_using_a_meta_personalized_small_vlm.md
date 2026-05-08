# Small-Large Collaboration: Training-efficient Concept Personalization for Large VLM using a Meta Personalized Small VLM

Sihan Yang, Huitong Ji, Shaolin Lu, Jiayi Chen, Binxiao Xu, Ming Lu, Yuanxing Zhang, Wenhui Dong, Wentao Zhang (2025)

## 🧩 Problem to Solve

본 논문은 Vision-Language Models (VLMs)를 사용자의 특정 개념(예: 반려동물, 특정 캐릭터 등)에 맞게 개인화(Personalization)하는 과정에서 발생하는 **막대한 학습 비용**과 **모델 접근성** 문제를 해결하고자 한다.

현재의 VLM 개인화 방식은 주로 파인튜닝(Fine-tuning)에 의존하는데, 이는 모델의 크기가 커질수록 연산 비용이 기하급수적으로 증가하는 문제를 야기한다. 또한, GPT-4o와 같은 최신 고성능 VLM들은 폐쇄형 API 형태로 제공되어 사용자가 직접 가중치를 수정하여 개인화하는 것이 불가능하다. 반면, 소형 VLM(Small VLM)은 학습 및 배포가 용이하지만 복잡한 추론 능력이 부족하여 단독으로는 고품질의 응답을 생성하기 어렵다.

따라서 본 연구의 목표는 소형 VLM의 효율적인 개인화 능력과 대형 VLM의 강력한 추론 능력을 결합하여, 학습 비용을 획기적으로 줄이면서도 폐쇄형 모델까지 지원 가능한 효율적인 개인화 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"역할의 명확한 분리(Division of Tasks)"**이다. 소형 VLM($M_s$)은 사용자 정의 개념에 대한 **지각(Perception)**과 검출을 담당하고, 대형 VLM($M_l$)은 이를 바탕으로 한 **반추(Reflection)**와 **일반 추론(General Reasoning)**을 담당하게 한다.

이를 위해 다음과 같은 핵심 설계 요소를 도입하였다:

1. **Meta Personalized Small VLM**: 모든 새로운 개념에 대해 매번 학습하는 대신, 오프라인에서 미리 학습된 메타 개념 어댑터 풀(Pool)을 구축하고, 추론 시점에 가장 유사한 어댑터를 선택하여 사용하는 Tuning-free 방식을 제안한다.
2. **Test-time Reflection Strategy**: 소형 VLM이 발생시킬 수 있는 환각(Hallucination)을 억제하기 위해, 대형 VLM이 소형 VLM의 검출 결과를 다시 한번 검증하는 2단계 Yes/No VQA 체크 메커니즘을 도입하였다.

## 📎 Related Works

기존의 VLM 개인화 연구들은 크게 세 가지 방향으로 진행되었다:

- **파인튜닝 기반 (Fine-tuning based)**: MyVLM, Yo’LLaVA 등이 있으며, 소프트 프롬프트나 외부 헤드를 학습시킨다. 하지만 모델 크기에 비례해 학습 비용이 급증한다는 한계가 있다.
- **RAG 기반 (Retrieval-Augmented Generation)**: RAP-LLaVA와 같이 외부 지식을 검색하여 보완하는 방식이나, 여전히 상당한 학습 비용이 소요된다.
- **표현 기반 (Representation-based)**: 개념을 임베딩 형태로 삽입하는 방식이다.

최근 소형-대형 모델 협업(Small-Large Collaboration) 연구가 진행되고 있으나, 대부분은 추론 비용 감소에 초점을 맞추고 있다. 본 논문은 이를 '개인화' 영역으로 확장하여 특히 **학습 비용(Training Cost)**의 병목 현상을 해결하려 한다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인 (SLC Inference Pipeline)

SLC 프레임워크는 다음의 세 단계로 구성된다:

1. **Test-time Detection**: 메타 학습된 소형 VLM($M_s$)이 이미지 $I_t$에서 등록된 사용자 개념 $C^u_i$를 검출하고, 구조화된 큐(Cue) $R_t$를 생성한다. $R_t$에는 개념의 존재 여부(`present`), 절대 위치(`loc_abs`), 상대 위치(`loc_rel`)가 포함된다.
2. **Test-time Reflection**: 대형 VLM($M_l$)이 $M_s$가 생성한 $R_t$를 바탕으로 자가 VQA 검증을 수행하여 환각을 제거한 정제된 큐 $\tilde{R}_t$를 생성한다.
3. **Answer Generation**: 최종적으로 $M_l$이 이미지, 질문, 그리고 정제된 큐 $\tilde{R}_t$를 모두 입력받아 최종 답변 $a_t$를 생성한다.

### 2. Meta-Personalized Small VLM

학습 비용을 줄이기 위해 $M_s$는 다음과 같은 메타 학습 과정을 거친다.

**오프라인 학습 (Offline Training):**

- 공개 데이터셋의 이미지들을 CLIP 임베딩으로 변환 후, K-means 클러스터링을 통해 $K$개의 메타 개념(Meta-concepts) $\left\{C^m_k\right\}_{k=1}^K$를 정의한다.
- 각 클러스터의 중심점을 기반으로 각각의 LoRA 어댑터 $A^m_k$를 학습시켜 오프라인 딕셔너리에 저장한다.

**온라인 추론 (Online Inference):**

- 새로운 사용자 개념 $C^u_i$가 입력되면, 참조 이미지들의 특징 평균을 내어 시나리오 임베딩 $\bar{e}_u$를 구한다.
- 코사인 유사도를 이용하여 가장 적합한 메타 어댑터 $k^\star$를 선택한다:
$$k^\star = \arg \max_k \cos(\bar{e}_u, e^m_k)$$
- 선택된 $A^{m}_{k^\star}$를 $M_s$에 플러그인하여 튜닝 없이 즉시 사용한다.

### 3. Test-Time Reflection of Large VLM

소형 VLM의 낮은 신뢰도를 보완하기 위해 대형 VLM이 수행하는 검증 절차이다.

1. **Identity Extraction**: $M_l$은 개념 설명 $T_{C^u_i}$에서 변하지 않는 정체성 구절 $ID(C^u_i)$ (예: "토끼 모양의 애니메이션 캐릭터")를 추출한다.
2. **Self-VQA Verification**: 추출된 $ID$와 $M_s$가 보고한 위치 정보를 결합해 두 가지 질문을 던진다:
   - $Q_1$: "이미지의 $\text{loc}_{\text{abs}}$ 위치에 $ID(C^u_i)$가 있는가?"
   - $Q_2$: "$ID(C^u_i)$가 $\text{loc}_{\text{rel}}$ 상태인가?"
3. **Cue Update**: 두 질문 모두 'No'인 경우 해당 개념은 존재하지 않는 것으로 판단하여 $\text{present}=0$으로 수정하며, 각각의 질문 결과에 따라 위치 정보를 삭제하거나 유지한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MC-LLaVA (다중 개념), Yo’LLaVA (단일 개념) 및 과적합 측정용 SQA(Special Question-Answer) 세트를 사용하였다.
- **비교 대상**: Yo’LLaVA, MC-LLaVA, RAP-LLaVA, GPT-4o (Upper bound), LLaVA-1.5-13B (Lower bound).
- **측정 지표**: 인식 정확도(Rec.), VQA 정확도, Text-only QA 정확도 및 학습 비용(FLOPs).

### 주요 결과

- **성능 및 효율성**: SLC (with GPT-4o)는 거의 모든 지표에서 최고 성능을 기록하였다. 특히 학습 비용 측면에서 SLC는 $1.7 \times 10^{17}$ FLOPs를 소모하여, Yo’LLaVA나 MC-LLaVA 대비 약 40배, RAP-LLaVA 대비 약 200배 더 효율적이다.
- **환각 억제 (SQA 결과)**: 단순 암기나 과적합을 테스트하는 SQA 세트에서 SLC는 GPT-4o와 대등한 수준의 높은 점수(0.900)를 기록했으며, 기존 파인튜닝 기반 방법들보다 10%p 이상 높은 성능을 보였다. 이는 Test-time Reflection이 환각을 효과적으로 제거했음을 입증한다.
- **확장성 (Scaling)**: 소형 모델($M_s$)의 크기가 커질수록, 그리고 대형 모델($M_l$)의 성능이 좋을수록 전체 시스템의 성능이 단조 증가하는 경향을 보였다.

## 🧠 Insights & Discussion

본 논문은 VLM 개인화에 있어 **"학습 효율성"**과 **"추론 정확도"**라는 두 마리 토끼를 잡기 위해 전략적인 협업 구조를 설계하였다.

**강점:**

- **Tuning-free 개인화**: 메타 어댑터 풀을 활용함으로써 새로운 사용자가 유입될 때마다 모델을 다시 학습시킬 필요가 없다.
- **모델 독립성**: 대형 VLM을 Frozen 상태로 사용하므로, 오픈소스 모델뿐만 아니라 API 기반의 폐쇄형 모델(GPT-4o 등)에도 즉시 적용 가능하다.
- **상호 보완적 시너지**: $M_s$는 정밀한 위치 큐를 제공하고, $M_l$은 이를 검증함으로써 단독 모델이 가질 수 없는 신뢰성을 확보하였다.

**한계 및 논의사항:**

- **메타 개념의 대표성**: K-means로 정의된 10개의 메타 개념이 세상의 모든 시각적 범주를 충분히 대표할 수 있는지에 대한 의문이 남는다. (다만, 실험 결과 10개에서 성능이 피크를 찍는 것으로 나타났다.)
- **추론 지연 시간 (Latency)**: $M_s$의 검출 $\rightarrow$ $M_l$의 반추 $\rightarrow$ 최종 생성으로 이어지는 다단계 파이프라인은 단일 모델 추론보다 시간이 더 소요될 가능성이 크다.

## 📌 TL;DR

본 논문은 소형 VLM($M_s$)과 대형 VLM($M_l$)의 역할을 분리하여 학습 비용을 획기적으로 낮춘 개인화 프레임워크 **SLC (Small-Large Collaboration)**를 제안한다. 소형 VLM은 오프라인에서 학습된 메타 어댑터를 통해 튜닝 없이 사용자 개념을 검출하고, 대형 VLM은 이를 다시 한번 검증(Reflection)하여 환각을 제거한 후 최종 답변을 생성한다. 이 방식은 학습 비용을 최대 200배까지 절감하면서도, 폐쇄형 모델까지 지원 가능하며 기존 파인튜닝 방식보다 뛰어난 인식 및 추론 성능을 보여준다.
