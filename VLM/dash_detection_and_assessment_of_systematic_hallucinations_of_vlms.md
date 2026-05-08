# DASH: Detection and Assessment of Systematic Hallucinations of VLMs

Maximilian Augustin, Yannic Neuhaus, Matthias Hein (2025)

## 🧩 Problem to Solve

본 논문은 Vision-Language Models(VLMs)에서 빈번하게 발생하는 **Object Hallucination**, 특히 이미지에 존재하지 않는 객체가 있다고 잘못 판단하는 **False-Positive(FP) hallucination** 문제를 해결하고자 한다.

기존의 hallucination 평가 벤치마크(예: POPE, AMBER)는 MSCOCO와 같은 소규모의 큐레이션된 데이터셋에 의존하고 있다. 이러한 접근 방식은 두 가지 핵심적인 한계를 가진다. 첫째, 실제 VLMs가 사용되는 개방형 환경(Open-world settings)에서 발생하는 다양한 hallucination을 평가하기에는 데이터의 다양성이 부족하다. 둘째, 모델이 특정 유형의 이미지에 대해 반복적으로 오류를 범하는 '체계적 오류(Systematic errors)'를 탐지하고 분석하는 체계적인 방법론이 부재하다.

따라서 본 연구의 목표는 대규모 실세계 이미지 데이터셋을 활용하여 VLMs의 체계적인 FP-hallucination을 자동으로 탐지, 평가하고 이를 완화할 수 있는 대규모 파이프라인인 **DASH**를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 VLMs가 어떤 시각적 패턴에 취약하여 hallucination을 일으키는지 체계적으로 찾아내는 자동화된 파이프라인을 설계한 것이다.

1. **DASH 파이프라인 제안**: 인간의 레이블링 없이도 체계적인 FP-hallucination을 식별하는 대규모 자동화 파이프라인을 구축하였다. 이는 텍스트 기반 쿼리를 생성하는 **DASH-LLM**과 이미지 기반 쿼리를 최적화하는 **DASH-OPT**로 구성된다.
2. **DASH-OPT의 최적화 기법**: Latent Diffusion Model의 생성 과정을 최적화하여, VLM은 "Yes"라고 답하지만 객체 탐지기(Object Detector)는 객체가 없다고 판단하는 '기만적 이미지'를 생성하는 방법론을 제시하였다.
3. **대규모 분석 및 벤치마크 구축**: PaliGemma 및 LLaVA-NeXT 모델을 대상으로 380개 객체 클래스에 대해 분석하여 19,000개 이상의 hallucination 클러스터를 발견하였으며, 이를 바탕으로 더 정밀한 평가를 위한 새로운 벤치마크인 **DASH-B**를 제안하였다.
4. **Hallucination 완화 가능성 증명**: DASH를 통해 수집한 모델 특화 데이터셋으로 PaliGemma를 미세 조정(Fine-tuning)함으로써 객체 hallucination을 유의미하게 줄일 수 있음을 보였다.

## 📎 Related Works

### VLMs의 Hallucination 벤치마크 및 완화

기존의 CHAIR, POPE, AMBER 등은 객체의 존재 여부를 묻는 Type II hallucination을 주로 평가하였다. 그러나 이러한 연구들은 MSCOCO와 같이 제한된 객체 클래스를 가진 데이터셋을 사용하므로, 실제 환경에서의 다양성을 반영하지 못하며 이미 많은 모델이 높은 성능을 보여 벤치마크가 포화(Saturated) 상태에 이르렀다는 한계가 있다. 완화 방법으로는 Contrastive Decoding이나 Instruction Tuning 등이 제안되었으나 체계적인 탐지 방법론과는 거리가 있다.

### 이미지 분류의 Spurious Correlations

이미지 분류 모델이 배경(예: 소와 풀밭)과 같은 가짜 상관관계(Spurious features)에 의존하여 오분류하는 문제는 이미 알려져 있다. 본 논문은 이러한 개념을 VLM의 hallucination으로 확장하여, 특정 배경이나 관련 객체가 나타날 때 VLM이 대상 객체를 잘못 인식하는 현상을 체계적으로 분석하고자 한다.

### 가이드 기반 이미지 생성(Guided Image Generation)

Stable Diffusion과 같은 모델을 이용해 모델을 기만하는 이미지를 생성하거나 모델의 취약점을 찾는 디버깅 기법들이 존재한다. DASH-OPT는 이러한 아이디어를 채택하여 자연 이미지 매니폴드(Natural image manifold) 상에서 VLM을 오작동하게 만드는 최적화된 쿼리 이미지를 생성한다.

## 🛠️ Methodology

DASH는 특정 객체 클래스에 대해 VLM이 반복적으로 hallucination을 일으키는 이미지 집합(Cluster)을 찾는 파이프라인이다. 전체 과정은 **쿼리 생성 $\rightarrow$ 탐색(Exploration) $\rightarrow$ 이용(Exploitation) $\rightarrow$ 클러스터링** 순으로 진행된다.

### 1. 쿼리 생성 (DASH-LLM & DASH-OPT)

VLM을 기만하기 위한 두 가지 경로의 쿼리를 생성한다.

* **DASH-LLM**: LLM(Llama 3.1-70B)을 사용하여 대상 객체와 자주 함께 나타나지만 객체 자체는 포함되지 않은 '가짜 특징(Spurious features)'을 묘사하는 50개의 텍스트 프롬프트를 생성한다.
* **DASH-OPT**: VLM의 취약점을 직접 공략하는 이미지를 생성한다. 단일 단계 확산 프로세스(Single-step diffusion process)의 잠재 변수를 최적화하며, 다음의 손실 함수를 최소화하는 조건 $C$를 찾는다.
  * **VLM 손실($\mathcal{L}_{vlm}$)**: VLM이 "Can you see a OBJ in this image?"라는 질문에 "Yes"라고 답할 확률을 최대화한다.
        $$\mathcal{L}_{vlm}(C) = -\log p_{vlm}(\text{"Yes"} | q(C), q_{stnOBJ})$$
  * **탐지기 손실($\mathcal{L}_{det}$)**: 개방형 객체 탐지기(OWLv2)가 해당 객체를 발견할 확률을 최소화하여, 실제로 객체가 생성되는 것을 방지한다.
        $$\mathcal{L}_{det}(C) = -\log(1 - p_{det}(OBJ | q(C)))$$
  * **최종 목적 함수**: $$\min_{C} \mathcal{L}_{vlm}(C) + \mathcal{L}_{det}(C)$$

### 2. 탐색 및 이용 단계 (Exploration & Exploitation)

생성된 쿼리를 사용하여 실제 웹 스케일 데이터셋인 ReLAION-5B에서 hallucination 유발 이미지를 찾는다.

* **Exploration (탐색)**: 텍스트 및 이미지 쿼리를 기반으로 CLIP kNN-retrieval을 수행하여 각 쿼리당 20장, 총 1,000장의 이미지를 추출한다. 이후 객체 탐지기로 객체가 실제로 있는 이미지를 제거하고, VLM이 "Yes"라고 답하는 FP-hallucination 이미지만 남긴다.
* **Exploitation (이용)**: 탐색 단계에서 발견된 성공적인 이미지들을 쿼리로 삼아 다시 kNN-retrieval을 수행한다. 이를 통해 단순한 오류가 아닌, 시각적으로 유사한 이미지들이 공통적으로 VLM을 속이는 '체계적 취약점'을 확장하여 찾는다.

### 3. 클러스터링 (Clustering)

수집된 이미지들을 DreamSim 거리 및 Agglomerative Clustering을 이용해 시각적으로 유사한 그룹으로 묶는다. 이를 통해 "특정 배경이나 특정 색상 조합이 나타나면 VLM이 해당 객체를 hallucinate 한다"는 결론을 도출할 수 있는 클러스터를 형성한다.

## 📊 Results

### 실험 설정

* **대상 모델**: PaliGemma, LLaVA-NeXT (Vicuna 및 Mistral 변형).
* **객체 클래스**: 총 380개 (COCO, Objects365, ImageNet, OpenImages에서 추출).
* **데이터셋**: ReLAION-5B.

### 주요 결과

1. **DASH-LLM vs DASH-OPT**: DASH-OPT가 DASH-LLM보다 더 많은 수의 이미지와 클러스터를 발견하였다. (PaliGemma 기준: LLM 1,892개 클러스터 $\rightarrow$ OPT 3,895개 클러스터). 특히 DASH-OPT는 LLM이 예측하지 못한 '알 수 없는 취약점(Unknown unknowns)'을 더 많이 찾아내며, 쿼리 이미지의 다양성(CLIP 거리 기준) 또한 더 높게 나타났다.
2. **취약점 전이(Transfer)**: 한 모델에서 발견된 체계적 hallucination 이미지가 다른 VLM에서도 동일하게 작동하는지 확인한 결과, 높은 전이율을 보였다. 특히 Vision Encoder(CLIP vs SigLIP)와 LLM Backbone의 종류가 hallucination 취약성에 큰 영향을 미쳤으며, 모델의 크기(Scale)는 hallucination 발생률 자체에는 적은 영향을 주었으나 탐지-환각 간의 트레이드오프를 개선하는 경향을 보였다.
3. **DASH-B 벤치마크**: 기존 POPE 벤치마크는 True Negative Rate(TNR)가 95% 이상으로 포화 상태인 반면, DASH-B에서는 모델들의 TNR이 현저히 낮게 측정되어(약 40%~70%), 현재의 VLM들에게 훨씬 더 어려운 정밀한 평가 도구임을 입증하였다.
4. **Fine-tuning 효과**: DASH를 통해 수집한 FP-hallucination 이미지(Negative)와 실제 객체 이미지(Positive)로 PaliGemma를 미세 조정했을 때, DASH-B의 정확도가 11.6%p 향상되는 등 hallucination 완화 효과가 확인되었다.

## 🧠 Insights & Discussion

본 연구는 VLM의 hallucination이 단순한 무작위 오류가 아니라, 특정 시각적 패턴에 결합된 **체계적인 취약점**임을 밝혀냈다.

**강점 및 통찰**:

* **웹 스케일 탐색의 필요성**: 큐레이션된 소규모 데이터셋으로는 발견할 수 없는 구체적인 취약점(예: 특정 브랜드의 신발을 사과로 인식하는 경우)을 ReLAION-5B와 같은 대규모 데이터셋 탐색을 통해 찾아낼 수 있었다.
* **시각적 컨텍스트의 오염**: VLM이 객체의 물리적 형태뿐만 아니라, 객체와 함께 자주 등장하는 환경(예: 산악 지형 $\rightarrow$ Ptarmigan 새)이나 문화적 맥락(예: 독일 전통 장식 $\rightarrow$ Baumkuchen 케이크)을 객체 자체로 오인하고 있음을 정성적으로 분석하였다.

**한계 및 비판적 해석**:

* **탐지기의 의존성**: 파이프라인이 OWLv2 객체 탐지기에 의존하고 있어, 탐지기가 객체를 놓치는 경우(False Negative) 이를 hallucination으로 잘못 분류할 위험이 있다. 저자들은 보수적인 임계값을 설정하여 이를 최소화하려 했으나, 완전히 제거할 수는 없다.
* **포괄성 문제**: ReLAION-5B가 매우 크지만 모든 자연 이미지의 분포를 대변하지는 못하므로, 모든 체계적 hallucination을 전수 조사하는 것은 불가능하다.

## 📌 TL;DR

본 논문은 VLMs가 특정 이미지 패턴에 대해 반복적으로 엉뚱한 객체를 인식하는 '체계적 hallucination'을 자동으로 탐지하는 **DASH** 파이프라인을 제안한다. LLM의 지식과 확산 모델의 최적화를 결합해 VLM을 속이는 쿼리를 만들고, 이를 통해 수십만 장의 hallucination 유발 이미지와 수만 개의 취약점 클러스터를 발견하였다. 또한, 이를 통해 기존 벤치마크의 한계를 지적하는 새로운 벤치마크 **DASH-B**를 제시하고, 해당 데이터를 이용한 미세 조정이 hallucination을 줄일 수 있음을 증명하였다. 이 연구는 향후 VLM의 신뢰성을 높이기 위한 데이터 구축 및 평가 프레임워크로서 중요한 역할을 할 것으로 기대된다.
