# Explainability for Vision Foundation Models: A Survey

Rémi Kazmierczak, Eloïse Berthier, Goran Frehse, Gianni Franchi (2025)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전 분야에서 급격히 성장하고 있는 **Vision Foundation Models (VFMs)**의 불투명성 문제를 해결하기 위한 **eXplainable AI (XAI)** 기술들의 현황을 분석한다.

최근의 AI 모델들은 파라미터 수가 기하급수적으로 증가하며 성능은 향상되었으나, 내부 의사결정 과정이 복잡해짐에 따라 '블랙박스'화 되는 경향이 있다. 특히 Foundation Model은 광범위한 데이터로 학습되어 일반화 성능이 뛰어나지만, 그 복잡성으로 인해 해석이 매우 어렵다. 흥미로운 점은 이러한 Foundation Model이 해석의 대상(Target)인 동시에, 다른 모델을 설명하기 위한 도구(Tool)로도 사용된다는 이중적 위치에 있다는 것이다. 

따라서 본 연구의 목표는 Vision Foundation Model과 XAI의 교차 지점에 있는 최신 연구들을 체계적으로 수집하고, 이를 분류 체계(Taxonomy)에 따라 분석하며, 현재의 한계점과 향후 연구 방향을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **포괄적인 문헌 수집 및 체계화**: XAI와 Vision Foundation Model의 접점에 있는 122편의 논문을 수집하여 분석했다.
2. **XAI 방법론의 분류 체계(Taxonomy) 제안**: Foundation Model을 활용한 XAI 기법을 크게 **Inherently Explainable Models (Ante-hoc)**와 **Post-hoc Methods**로 구분하고, 이를 다시 세부 카테고리(CBM, CoT, Rationale Generation 등)로 세분화하였다.
3. **평가 지표 및 공리(Axioms) 분석**: XAI 모델의 품질을 평가하기 위한 5가지 핵심 공리(Trustworthiness, Robustness, Complexity, Generalizability, Objectiveness)를 정의하고, 현재 PFM 기반 XAI 연구들의 정량적 평가 부족 문제를 지적했다.
4. **PFM 특화 챌린지 식별**: Transformer 기반 아키텍처로의 전환에 따른 Saliency map의 한계, spurious correlation(가짜 상관관계), 추론 능력의 한계 등 PFM 시대의 새로운 문제점들을 정의했다.

## 📎 Related Works

기존의 XAI 서베이들은 주로 일반적인 DNN이나 특정 도메인(예: 의료 이미지)의 해석 가능성에 집중했다. 최근에는 LLM(Large Language Models)의 설명 가능성에 대한 연구가 활발히 진행되었으나, 비전-언어 멀티모달 모델(VLM)이나 세그멘테이션 모델(SAM 등)과 같은 **Vision Foundation Model에 특화된 종합적인 서베이는 부족한 실정**이다.

본 논문은 기존의 XAI 분류 체계를 계승하면서도, 특히 **Pretrained Foundation Models (PFMs)**가 어떻게 XAI의 구성 요소로 통합되는지에 초점을 맞춘다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

본 논문은 수집된 논문들을 기반으로 PFM을 활용한 XAI 방법론을 다음과 같은 구조로 분석한다.

### 1. Inherently Explainable Models (Ante-hoc)
모델 설계 단계부터 해석 가능성을 내장한 방법론이다.

*   **Concept Bottleneck Models (CBM)**: 
    입력 데이터를 바로 결과로 연결하지 않고, 중간에 인간이 이해할 수 있는 '개념(Concept)' 층을 둔다. 
    $$\text{Input} \rightarrow \text{Concept Predictor} \rightarrow \text{Concepts} \rightarrow \text{Classifier} \rightarrow \text{Output}$$
    PFM(예: CLIP)의 멀티모달 일반화 능력을 사용하여, 별도의 대규모 레이블링 데이터 없이도 의미 있는 개념 임베딩을 생성한다.
*   **Textual Rationale Generation**: 
    모델의 예측과 함께 "왜 이런 예측을 했는가"에 대한 텍스트 근거를 생성한다. 주로 LLM이나 VLM(LLaVA, BLIP-2 등)을 디코더로 활용하여 자연어 설명을 출력한다.
*   **Chain of Thought (CoT) Reasoning**: 
    추론 과정을 여러 개의 해석 가능한 단계로 분해하여 순차적으로 처리한다. 이는 인간의 사고 과정과 유사하게 단계별 논리 전개를 보여줌으로써 투명성을 높인다.
*   **Prototypical Networks**: 
    학습 과정에서 대표적인 '프로토타입' 벡터들을 학습하고, 입력 데이터가 어떤 프로토타입과 유사한지를 통해 결과를 도출한다.

### 2. Post-hoc Explanation Methods
학습이 완료된 모델을 수정하지 않고 외부에서 설명을 추출하는 방법론이다.

*   **Input Perturbation**: 
    입력 데이터에 미세한 변화를 주어 모델 출력의 변화를 관찰함으로써 어떤 특징(Feature)이 중요한지 분석한다. (예: LIME, SHAP). 최근에는 SAM과 같은 PFM을 이용해 의미 있는 세그멘테이션 단위로 섭동을 주는 방식이 사용된다.
*   **Counterfactual Examples**: 
    "만약 $\text{X}$가 아니라 $\text{X}'$였다면 결과가 바뀌었을 것인가?"라는 가정을 통해 인과관계를 설명한다.
    $$do(Z = z^0 + \epsilon)$$
    여기서 $\epsilon$은 예측 결과(라벨)를 바꾸기 위한 최소한의 섭동을 의미하며, Stable Diffusion과 같은 생성 모델을 통해 시각적으로 이해 가능한 반사실적 이미지를 생성한다.
*   **Meta Explanation Datasets**: 
    특정 편향(Bias)을 유도하도록 설계된 보조 데이터셋을 사용하여 모델의 전역적인 동작 특성을 통계적으로 분석한다.
*   **Neuron/Layer Interpretation**: 
    특정 뉴런이나 레이어를 강하게 활성화하는 입력 패턴을 최적화 기법으로 찾아내어, 해당 뉴런이 어떤 시각적 특징에 반응하는지 분석한다.

## 📊 Results

본 논문은 개별 실험 결과보다는 수집된 122편의 논문에 대한 통계적 분석과 경향성을 제시한다.

*   **정량적 평가의 부족**: 분석 결과, PFM 기반 XAI 연구 중 단 **36%만이 정량적 평가 결과**를 포함하고 있었다. 이는 기존 XAI 연구들의 정량 평가 비율(약 58%)보다 현저히 낮은 수치이다.
*   **평가 지표의 편중**: 텍스트 생성 기반의 설명은 BLEU, ROUGE 등의 텍스트 유사도 지표를 주로 사용하며, 반사실적 이미지 생성은 FID나 LPIPS 같은 이미지 품질 지표에 의존하는 경향이 있다.
*   **PFM 도입의 효과**: PFM을 도입함으로써 과거 CBM 등이 겪었던 '특수 데이터셋에 대한 의존성'이 사라졌으며, 모델 전체를 재학습시키지 않고 Frozen PFM 상단에 가벼운 프로브(Probe)를 붙이거나 LoRA 등을 이용한 부분 학습만으로도 높은 해석 성능을 얻을 수 있음을 확인했다.

## 🧠 Insights & Discussion

### 강점 및 기회
PFM은 고엔트로피의 이미지 데이터를 저엔트로피의 임베딩 공간으로 변환하는 능력이 탁월하다. 저자들은 PFM 자체의 내부 구조를 이해하려 하기보다, PFM이 생성하는 **잠재 공간(Latent Space)의 분포를 모델링**하는 것이 더 현실적이고 효과적인 XAI 방향이 될 수 있다고 주장한다.

### 한계 및 비판적 해석
1. **기능성과 수학적 근거의 충돌**: 최신 PFM 기반 XAI는 "텍스트로 설명해주니 이해하기 쉽다"는 기능적 측면(Functionality)에 치중하고 있으며, SHAP처럼 수학적 이론에 기반한 엄밀한 근거(Mathematical Grounding)는 부족한 실정이다.
2. **가짜 설명(Spurious Explanations)의 위험**: PFM 자체가 학습 데이터의 편향(Bias)을 가지고 있기 때문에, XAI 모델이 내놓는 설명 또한 모델의 실제 추론 경로가 아니라 PFM이 가진 사전 지식에 기반한 '그럴듯한 거짓말'일 가능성이 크다.
3. **인간 인지와의 유사성 착각**: CoT와 같은 기법이 인간의 사고방식을 닮았다고 해서 그것이 곧 실제 모델의 작동 방식과 일치한다는 보장은 없다.

## 📌 TL;DR

본 논문은 Vision Foundation Model(PFM)을 활용한 XAI 기법들을 **Ante-hoc(내장형)**과 **Post-hoc(사후분석형)**으로 체계화하여 분석한 종합 서베이이다. PFM의 도입으로 데이터 의존성은 줄고 설명의 형태는 다양해졌으나, **정량적 평가의 부족**과 **수학적 근거 결여**, 그리고 **가짜 설명 생성 가능성**이라는 심각한 과제를 안고 있다. 향후 연구는 PFM의 잠재 공간 모델링과 멀티모달 통합 평가 지표 개발에 집중해야 한다.