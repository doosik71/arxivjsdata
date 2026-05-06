# Self-Supervised Visual Preference Alignment

Ke Zhu, Liang Zhao, Zheng Ge, Xiangyu Zhang (2024)

## 🧩 Problem to Solve

본 논문은 Vision-Language Models(VLMs)에서 사용자의 의도(user-intention)와 모델의 출력을 일치시키는 **Preference Alignment(선호도 정렬)** 문제를 다룬다. 기존의 LLaVA와 같은 VLM들은 대규모 이미지-텍스트 쌍으로 사전 학습(pretraining)되고 지도 미세 조정(SFT) 과정을 거치지만, 여전히 환각(hallucination) 현상이 발생하거나 사용자의 복잡한 지시사항을 제대로 따르지 못하는 한계가 있다.

이러한 문제를 해결하기 위해 RLHF(Reinforcement Learning from Human Feedback)나 DPO(Direct Preference Optimization)와 같은 정렬 기술이 제안되었으나, VLM 분야에서는 다음과 같은 문제점이 존재한다. 첫째, 선호도 데이터 구축을 위해 GPT-4와 같은 강력한 외부 모델이나 사람이 직접 라벨링하는 고비용의 데이터가 필요하다. 둘째, 기존의 시도들은 특정 태스크 도메인에 한정되어 범용적인 성능 향상을 이끌어내는 데 한계가 있다. 따라서 본 논문의 목표는 외부의 정답 라벨이나 인간의 개입 없이, **자기 지도 학습(Self-supervised)** 방식으로 선호도 데이터를 생성하고 모델을 정렬하는 효율적인 파이프라인을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **이미지 증강(Image Augmentation)을 통해 모델이 스스로 '틀린 답(Hard Negative)'을 생성하게 유도하고, 이를 통해 정렬을 수행**하는 것이다.

구체적으로, 원본 이미지에 대한 모델의 응답을 '선택된 응답(chosen response)'으로, 적절히 설계된 증강(augmentation)이 적용된 이미지에 대한 모델의 응답을 '거절된 응답(rejected response)'으로 정의한다. 이미지에 약간의 왜곡을 가하면 VLM은 의미론적으로는 비슷하지만 사실적으로는 틀린 응답을 생성할 가능성이 높으며, 이 두 응답의 쌍을 DPO 학습에 활용함으로써 모델이 더 강건하고 정확한 답변을 생성하도록 유도할 수 있다.

## 📎 Related Works

**1. Large Vision-Language Models (VLMs):**
LLaVA, InstructBLIP 등은 시각 신호를 LLM에 정렬하여 멀티모달 이해 능력을 높였다. 대개 사전 학습 후 SFT 단계를 거치지만, 이러한 방식만으로는 사용자의 의도에 완전히 정렬되지 않아 환각 현상이 지속되는 문제가 있다.

**2. Preference Alignment in LLM/VLM:**
RLHF, DPO, PPO 등이 LLM의 독성 출력 감소 및 의도 정렬을 위해 사용되었다. 최근 VLM에서도 DPO를 적용하려는 시도(예: HA-DPO)가 있었으나, 이는 환각 제거와 같은 특정 작업에 치중되어 있으며, 무엇보다 GPT-4나 인간의 피드백에 의존하는 데이터 구축 파이프라인으로 인해 확장성이 떨어진다는 한계가 있다.

**3. Contrastive Learning:**
이미지의 서로 다른 증강 뷰(augmented views)가 유사한 의미적 임베딩을 공유하도록 학습하는 자기 지도 학습 방식이다. 본 논문은 이와 유사하게 이미지 증강을 사용하지만, 임베딩 공간의 유사도가 아닌 **생성된 텍스트 응답의 선호도**를 최적화한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

SeVa(Self-supervised Visual preference alignment)의 전체 과정은 다음과 같다.

1. **데이터 샘플링:** 비라벨링 이미지-질문 쌍 $(I, q)$을 무작위로 선택한다.
2. **응답 생성:** SFT가 완료된 VLM $\pi^{SFT}$를 사용하여 두 가지 응답을 생성한다.
   - **Chosen response ($y_c$):** 원본 이미지 $I$를 입력하여 생성한 응답.
   - **Rejected response ($y_r$):** 이미지 증강 함수 $T$가 적용된 이미지 $T(I)$를 입력하여 생성한 응답.
3. **필터링:** $y_c$와 $y_r$이 동일한 경우 학습에 도움이 되지 않으므로 제외하고, 서로 다른 응답만 남겨 선호도 데이터셋 $D^{SeVa}$를 구성한다.
4. **DPO 학습:** 구성된 데이터셋을 사용하여 모델의 파라미터를 최적화한다.

### 주요 방정식 및 손실 함수

**1. 응답 생성 과정:**
$$y^{(j)}_c = \pi^{SFT}(g(I_j), q_j)$$
$$y^{(j)}_r = \pi^{SFT}(g(T(I_j)), q_j)$$
여기서 $g(\cdot)$는 비전 인코더와 프로젝션 레이어를 의미하며, $T(\cdot)$는 이미지 증강 함수이다.

**2. DPO 손실 함수:**
학습 시에는 다음과 같은 DPO 손실 함수 $\mathcal{L}_d$를 최소화하여 모델 $\pi_{\theta'}$를 업데이트한다.
$$\mathcal{L}_d = -\mathbb{E}_{D} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta'}(y_c|x)}{\pi_{ref}(y_c|x)} - \beta \log \frac{\pi_{\theta'}(y_r|x)}{\pi_{ref}(y_r|x)} \right) \right]$$

- $\pi_{ref}$는 기준 모델(일반적으로 $\pi^{SFT}$)이다.
- $\beta$는 KL 발산 제약 강도를 조절하는 하이퍼파라미터이다.
- $\sigma$는 시그모이드 함수이다.

### Contrastive Learning과의 관계

본 연구는 SeVa가 시각적 대조 학습(Visual Contrastive Learning)의 특수한 형태로 볼 수 있음을 수학적으로 제시한다. InfoNCE 손실 함수에서 단 하나의 부정 샘플(negative sample)만 사용하는 경우, DPO의 최적화 방향과 유사한 구조를 가진다는 점을 통해 SeVa의 이론적 근거를 마련하였다.

## 📊 Results

### 실험 설정

- **기반 모델:** LLaVA-1.5-7B 및 13B.
- **데이터셋:** LLaVA665k 중 TextVQA 및 OCRVQA 데이터셋에서 샘플링한 16k (필터링 후 약 8k) 쌍 사용.
- **증강 기법:** Diffusion noise(기본값), RandFlip, MOCO 등 다양한 기법 실험.
- **평가 지표:** MMVet, LLaVA-Bench, MMBench, POPE 등 9개 벤치마크.

### 주요 결과

- **정량적 성과:** SeVa-7B 모델이 MMVet에서 LLaVA-1.5-7B 대비 6.7% 향상되었으며, 심지어 LLaVA-1.5-13B의 성능을 일부 상회하는 결과를 보였다.
- **범용적 개선:** LLaVA-Bench-in-the-wild에서 SeVa-13B는 GPT-4 대비 상대 점수 80%를 달성하여 기존 LLaVA-1.5-13B보다 크게 향상된 성능을 보였다.
- **SFT와의 비교:** 동일한 양의 데이터를 사용한 지속적 SFT(continual SFT)보다 SeVa가 훨씬 적은 데이터와 학습 시간으로 더 높은 성능을 기록하였다(표 3 참고).
- **데이터 규모:** 샘플링 데이터 수를 2k에서 16k로 늘릴수록 성능이 일관되게 향상되어, 향후 데이터 스케일링 가능성을 확인하였다.

## 🧠 Insights & Discussion

**1. Hard Negatives의 중요성:**
실험 결과, 너무 약한 증강(RandFlip)이나 너무 강한 증강(Diffusion-Strong)보다 중간 정도의 왜곡을 주는 증강(Diffusion-Weak)이 가장 효과적이었다. 이는 모델이 정답과 매우 유사하지만 틀린 'Hard Negative' 응답을 생성했을 때, 이를 거절하는 방향으로 학습하는 것이 모델의 이해 능력을 가장 크게 향상시킨다는 것을 의미한다.

**2. 모델 캘리브레이션(Calibration) 효과:**
GPT-4를 이용한 일관성 테스트(Table 5) 결과, SeVa 모델은 높은 온도(Temperature) 설정에서도 응답의 일관성이 더 높게 유지되었다. 이는 SeVa가 잘못된 토큰의 생성 확률을 낮춤으로써 모델의 출력을 더 강건하게 만드는 캘리브레이션 효과를 가졌음을 시사한다.

**3. 정성적 분석:**
시각화 결과(Fig 6), SeVa는 다음과 같은 능력이 눈에 띄게 향상되었다.

- **OCR 능력:** 가스 가격과 같은 정확한 숫자를 인식하는 능력이 강화되었다.
- **환각 감소:** 레시피 단계와 같은 세부 공정을 정확히 이해하여 잘못된 정보를 제공하는 빈도가 줄었다.
- **세부 응답 생성:** 단순한 나열보다 더 구체적이고 유용한 정보를 제공하는 경향(Helpfulness)을 보였다.

**4. 한계 및 논의:**
전통적인 QA 벤치마크(SQA, GQA)에서는 약간의 성능 하락이 관찰되었다. 저자들은 이를 '전통적 QA의 지시 이행 능력'과 '현대적 VLM의 종합적 이해 능력' 사이의 트레이드-오프(trade-off)로 해석하며, 이는 현재 VLM 연구 분야의 공통적인 쟁점임을 언급한다.

## 📌 TL;DR

본 논문은 외부의 정답 라벨이나 고비용의 인간/GPT-4 피드백 없이, **이미지 증강을 통해 스스로 생성한 '틀린 답'을 거절하도록 학습시키는 자기 지도 방식의 선호도 정렬(SeVa)**을 제안한다. 이 방법은 구현이 매우 간단함에도 불구하고 LLaVA-1.5의 멀티모달 이해 능력, OCR 성능, 환각 억제 능력을 크게 향상시켰다. 특히 고비용의 데이터 구축 과정 없이도 모델을 정렬할 수 있다는 점에서, 향후 대규모 VLM의 효율적인 정렬 및 스케일링에 중요한 역할을 할 가능성이 높다.
