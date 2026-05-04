# INT: Instance-Specific Negative Mining for Task-Generic Promptable Segmentation

Jian Hu, Zixu Cheng, Shaogang Gong (2025)

## 🧩 Problem to Solve

본 논문은 **Task-generic promptable image segmentation**에서 발생하는 문제를 해결하고자 한다. 이 과업의 목표는 하나의 일반적인 태스크 설명(task-generic prompt, 예: "the polyp")만을 사용하여, 다양한 이미지 샘플 각각에 적합한 인스턴스별 프롬프트(instance-specific prompt)를 추론하고 이를 통해 세그멘테이션을 수행하는 것이다.

기존 방법들은 Vision-Language Models(VLMs)의 일반화 능력을 활용하여 일반 프롬프트를 인스턴스별 프롬프트로 변환하여 SAM(Segment Anything Model)과 같은 모델을 가이드한다. 그러나 VLM이 복잡한 이미지(예: 위장된 객체, 의료 영상 등)에서 일반화에 어려움을 겪을 경우, 잘못된 인스턴스별 프롬프트가 생성될 가능성이 높다. 특히 정답 라벨(ground-truth)이 없는 테스트 단계에서 한 번 잘못 생성된 프롬프트는 반복적인 최적화 과정에서도 오류가 누적되어 전파(error propagation)되는 심각한 문제가 발생한다.

결과적으로 본 논문은 별도의 추가 학습 없이, VLM의 출력을 이용해 잘못된 프롬프트를 적응적으로 제거하고 정확한 인스턴스별 프롬프트를 찾아내어 세그멘테이션 성능을 높이는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Instance-specific Negative Mining (INT)**이다. 이는 이미지의 특정 영역을 마스킹했을 때 VLM의 출력값이 얼마나 크게 변하는지를 관찰하여 프롬프트의 정확성을 평가하는 기법이다.

정확한 프롬프트에 해당하는 객체는 해당 영역이 가려졌을 때 VLM의 확신도가 급격히 떨어지지만, 잘못 예측된 클래스(negative)의 경우 가려지더라도 출력값의 변화가 적다는 점에 착안하였다. 이를 통해 정답 라벨 없이도 VLM의 출력 변화량을 측정함으로써 가장 가능성이 높은 인스턴스별 프롬프트를 선택하고, 반복적인 과정을 통해 부정적인(irrelevant) 지식의 영향력을 점진적으로 제거하는 **Progressive Negative Mining**을 제안한다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 다룬다.

- **Vision-Language Models (VLMs):** CLIP, LLaVA 등 이미지와 텍스트를 동시에 이해하는 모델들이다. 강력한 일반화 능력을 갖추고 있으나, 의료 영상과 같이 데이터가 희소한 도메인에서는 성능이 저하되는 한계가 있다.
- **Promptable Segmentation:** SAM과 같이 포인트, 박스, 텍스트 등의 프롬프트를 입력받아 객체를 분할하는 모델들이다. 하지만 대부분 수동 프롬프트에 의존하며, 이는 주관적이고 모호할 수 있다는 단점이 있다.
- **GenSAM 및 ProMaC:** 수동 프롬프트 없이 task-generic prompt를 사용하여 인스턴스별 프롬프트를 추론하려는 시도들이다. ProMaC는 환각(hallucination)을 사전 지식으로 활용하여 프롬프트를 최적화하지만, 정답 라벨이 없는 상황에서 생성된 프롬프트의 품질을 검증하고 오류를 수정할 메커니즘이 부족하여 오류 전파 문제에 노출되어 있다.

INT는 이러한 기존 방식과 달리, VLM 출력의 대비(contrast)를 이용한 네거티브 마이닝을 통해 반복적으로 프롬프트를 교정한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

INT는 학습이 필요 없는(training-free) 테스트 시간 적응(test-time adaptation) 방식으로, 크게 두 가지 구성 요소로 이루어져 있다.

### 1. Instance-Specific Prompt Generation

이 단계에서는 일반 프롬프트 $P_g$를 이미지 $X$에 최적화된 인스턴스별 프롬프트 $A^u_i$로 변환한다.

**가. Hallucination-driven Candidates Generation**
이미지를 다양한 스케일(원본, 절반, 4등분)의 패치로 나누어 VLM에 입력한다. 이는 객체가 일부만 보이거나 가려진 상황에서도 VLM이 사전 지식을 이용해 객체의 이름($A^{fore}, A^{back}$)과 바운딩 박스($B_k$)를 추론하도록 유도하기 위함이다.

**나. Prompt Selection with Negative Mining**
후보 프롬프트의 정확성을 검증하기 위해, Stable Diffusion 기반의 인페인팅 모듈($F_{in}$)을 사용하여 후보 영역을 제거한 이미지 $X'_k$를 생성한다. 이때 다음과 같은 수식으로 VLM 출력의 차이를 계산한다.

$$D(y^k_i) = \max(\text{softmax}(\text{logit}_\theta(y_t|X_k, P, y_{<t})) - \text{softmax}(\text{logit}_\theta(y_t|X'_k, P, y_{<t})))$$

여기서 $X_k$는 원본 패치, $X'_k$는 마스킹된 패치이다. 출력값의 차이가 가장 큰 카테고리를 해당 패치의 최종 예측값으로 선택하며, 전체 패치 중 가장 큰 변화를 보인 후보를 해당 반복 회차의 인스턴스별 프롬프트 $A^u_i$로 결정한다.

**다. Progressive Negative Mining**
단일 반복에서는 모호한 샘플로 인해 잘못된 클래스가 높은 차이를 보일 수 있다. 이를 해결하기 위해 반복 회차 간의 점수를 누적 곱하는 방식을 사용한다. 먼저 현재 회차의 차이값을 정규화한다.

$$D^{norm}(y^k_i) = \frac{D(y^k_i)}{\sum_{k=1}^K D(y^k_i)}$$

이후 다음 회차의 차이값에 이전 회차의 정규화된 값을 곱하여 업데이트한다.

$$D(y^k_{i+1}) = D(y^k_{i+1}) \cdot D^{norm}(y^k_i)$$

이 과정을 통해 일시적인 오류는 억제되고, 모든 회차에서 일관되게 높은 변화를 보이는 정답 카테고리의 신호만 증폭된다.

### 2. Semantic Mask Generation

생성된 $A^u_i$를 사용하여 정밀한 마스크를 생성하는 과정이다.

1. **객체 탐지:** GroundingDINO를 사용하여 $A^u_i$에 해당하는 모든 잠재적 바운딩 박스 $B^k_i$를 수집한다.
2. **마스크 생성:** Spatial CLIP을 통해 텍스트 프롬프트를 시각적 프롬프트로 변환하고, 이를 $B^k_i$와 함께 SAM에 입력하여 초기 마스크 $m^k_i$를 생성한다.
3. **의미적 정렬 (Semantic Alignment):** 생성된 마스크 영역과 텍스트 프롬프트 $A^u_i$ 간의 CLIP 유사도 $s(m^k_i)$를 계산하여, 가중합을 통해 최종 마스크 $M_i$를 산출한다.
   $$M_i = \sum_{k=1}^K (s(m^k_i) * m^k_i)$$
4. **이미지 업데이트:** 생성된 마스크를 이용해 다음 반복 회차에서 사용할 이미지 $X_{i+1}$을 업데이트하여 무관한 영역의 간섭을 줄인다.
   $$X_{i+1} = w \cdot (X_i \odot M_i) + (1-w) \cdot X_i \quad (w=0.3)$$
5. **최종 결과 도출:** 모든 반복 회차의 마스크들의 평균값과 가장 가까운 마스크 $M^*_i$를 최종 결과물로 선택한다.

## 📊 Results

### 실험 설정
- **데이터셋 및 작업:** 
    - Camouflaged Object Detection (COD): CHAMELEON, CAMO, COD10K
    - Medical Image Segmentation (MIS): CVC-ColonDB, Kvasir (폴립), ISIC (피부 병변)
- **비교 대상:** 
    - 약지도 학습 방법 (Scribble, Point supervision)
    - VLM+SAM 조합 (GPT4V+SAM, LLaVA1.5+SAM)
    - 최신 프롬프트 기반 세그멘테이션 (GenSAM, ProMaC, GroundingSAM 등)
- **지표:** Mean Absolute Error ($M$), adaptive F-measure ($F_\beta$), mean E-measure ($E_\phi$), structure measure ($S_\alpha$).

### 주요 결과
- **COD 작업:** INT는 단 하나의 일반 프롬프트만 사용함에도 불구하고, 포인트 기반 및 스크리블 기반의 약지도 학습 방법들보다 모든 데이터셋에서 우수한 성능을 보였다. 특히 ProMaC보다 높은 수치를 기록하며 state-of-the-art 성능을 달성하였다.
- **MIS 작업:** VLM이 의료 영상에 특화되어 학습되지 않았음에도 불구하고, INT의 네거티브 마이닝과 후보군 탐색 과정을 통해 폴립 및 피부 병변 세그멘테이션에서 기존 VLM+SAM 조합보다 월등한 성능 향상을 보였다.
- **Ablation Study:** 
    - **반복 횟수:** 약 5회 반복 이후 성능이 수렴하고 안정화됨을 확인하였다.
    - **이미지 전처리:** 원본, 절반, 4등분 패치를 모두 사용하는 전략이 전역 및 지역 정보를 모두 포착하여 가장 성능이 좋았다.
    - **모듈 기여도:** 특히 Progressive Negative Mining(PNM)을 제거했을 때 성능이 크게 하락하여, 초기 오류를 교정하는 PNM의 중요성이 입증되었다.

## 🧠 Insights & Discussion

본 논문은 VLM이 가진 '특징 소멸 시 예측값 급변'이라는 특성을 이용하여, 정답 라벨 없이도 프롬프트의 품질을 스스로 검증하는 메커니즘을 설계했다는 점에서 매우 영리한 접근을 보여준다. 

**강점:**
- **학습 불필요 (Training-free):** 추가적인 파라미터 업데이트 없이 테스트 단계에서 적응하므로 확장성이 매우 높다.
- **오류 교정 능력:** 단순히 프롬프트를 생성하는 것에 그치지 않고, 반복적인 네거티브 마이닝을 통해 잘못된 프롬프트를 걸러내는 자가 수정(self-correcting) 능력을 갖추었다.
- **범용성:** 자연 이미지(COD)뿐만 아니라 전문 지식이 필요한 의료 영상(MIS)에서도 효과적임을 증명하였다.

**한계 및 논의사항:**
- **연산 비용:** 여러 번의 반복 회차(epoch) 동안 VLM 추론, 인페인팅, SAM-CLIP 과정을 거쳐야 하므로 추론 시간이 상당히 소요될 것으로 예상된다.
- **하이퍼파라미터 의존성:** 이미지 업데이트 가중치 $w$나 반복 횟수 $I$와 같은 설정값이 성능에 영향을 줄 수 있다.
- **VLM 의존성:** 기본적으로 LLaVA-1.5와 같은 강력한 VLM의 성능에 기반하고 있으므로, VLM 자체가 완전히 인식하지 못하는 객체에 대해서는 한계가 있을 수 있다.

## 📌 TL;DR

본 논문은 일반적인 태스크 프롬프트에서 인스턴스별 프롬프트를 정확하게 추론하기 위해 **Progressive Negative Mining (INT)** 기법을 제안한다. 이는 객체 영역을 마스킹했을 때 VLM 출력값이 크게 변하는 특성을 이용해 정답 프롬프트를 찾아내고, 반복적인 누적 곱 연산을 통해 잘못된 프롬프트의 영향을 제거한다. 실험 결과, 위장 객체 탐지 및 의료 영상 분할과 같은 난이도 높은 작업에서 기존의 약지도 학습 및 VLM 기반 방법들을 뛰어넘는 성능을 보였다. 이 연구는 정답 라벨 없이도 VLM의 내부 상태 변화를 이용해 프롬프트를 최적화할 수 있음을 보여주어, 향후 제로샷/퓨샷 세그멘테이션 연구에 중요한 기여를 할 것으로 보인다.