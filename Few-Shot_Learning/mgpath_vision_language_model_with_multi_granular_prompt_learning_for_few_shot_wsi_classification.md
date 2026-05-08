# MGPATH: Vision-Language Model with Multi-Granular Prompt Learning for Few-Shot WSI Classification

Anh-Tien Nguyen et al. (2025)

## 🧩 Problem to Solve

본 논문은 전체 슬라이드 이미지(Whole Slide Image, WSI) 분류에서 발생하는 두 가지 핵심적인 난제를 해결하고자 한다. 첫째는 WSI가 기가픽셀 수준의 거대한 크기를 가지고 있어 정밀한 어노테이션을 수행하는 데 막대한 비용과 시간이 소요된다는 점이며, 이로 인해 학습 데이터가 부족한 Few-shot 상황에서 모델의 일반화 성능이 떨어진다는 것이다. 둘째는 기존의 Multiple Instance Learning (MIL) 및 Vision-Language Model (VLM) 방식들이 이미지 내의 계층적 구조나 국소적-전역적 특징 간의 복잡한 상관관계를 충분히 포착하지 못하며, 특히 단순한 Cosine Similarity 기반의 정렬 방식이 데이터 증강 과정의 섭동(perturbation)이나 텍스트-이미지 간의 불완전한 일치 문제에 취약하다는 점이다. 따라서 본 연구의 목표는 대규모 병리 이미지 사전 학습 모델을 효율적으로 활용하고, 다중 입도(multi-granular) 프롬프트 학습과 최적 운송(Optimal Transport, OT) 이론을 도입하여 적은 양의 데이터로도 높은 분류 성능을 내는 VLM 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 병리 이미지의 계층적 특성을 반영한 **Multi-granular Prompt Learning**과 데이터 분포의 이질성을 극복하기 위한 **Optimal Transport 기반의 정렬**이다. 구체적으로는 13억 개의 병리 패치로 학습된 Prov-GigaPath 비전 인코더와 PLIP 텍스트 인코더를 경량 Adaptor로 연결하여 파라미터 효율적인 VLM을 구성하였다. 또한, 개별 패치 수준의 주의 집중(Attention)과 공간적 그룹 수준의 주의 집중을 결합하여 미세한 세포 디테일과 광범위한 조직 맥락을 동시에 포착하도록 설계하였다. 마지막으로, 시각적 임베딩과 텍스트 임베딩 간의 거리를 측정할 때 단순 내적이 아닌 OT 거리를 사용하여 노이즈에 강건한 정렬을 구현하였다.

## 📎 Related Works

기존의 WSI 분석은 주로 MIL 방식에 의존해 왔으나, 질병 관련 패치가 매우 희소하게 분포하는 특성상 무관한 데이터에 의해 유용한 정보가 희석되는 문제가 있었다. 최근에는 CLIP과 같은 VLM이 등장하여 제로샷(Zero-shot) 학습 가능성을 보여주었으나, 일반 자연어 이미지로 학습된 CLIP은 병리 도메인의 특수성을 반영하지 못하며, PLIP나 CONCH 같은 병리 특화 VLM조차도 WSI의 거대한 구조적 맥락을 처리하는 데 한계가 있었다. 특히 기존의 프롬프트 학습 방식들은 주로 고정된 시각적 특징에 프롬프트를 접두사(prefix) 형태로 추가하는 방식에 그쳐, 이미지 내부의 공간적 관계나 계층적 구조를 무시하는 경향이 있었다. MGPATH는 이러한 한계를 극복하기 위해 공간 그래프 기반의 메시지 패싱과 다중 입도 어텐션을 도입하여 차별화를 꾀하였다.

## 🛠️ Methodology

### 1. Adaptor-based Vision-Text Alignment

본 모델은 Prov-GigaPath의 비전 인코더 $E^I(\cdot)$와 PLIP의 텍스트 인코더 $E^T(\cdot)$를 연결하기 위해 경량 Adaptor $A^I(\cdot)$와 $A^T(\cdot)$를 도입한다. 두 인코더의 출력 차원이 서로 다르므로, Adaptor를 통해 동일한 은닉 차원 $\mathbb{R}^d$로 투영한다. 학습은 약 923K 개의 병리 이미지-텍스트 쌍을 사용하여 Noise Contrastive Loss를 통해 수행하며, 백본 인코더들은 고정(frozen)시킨 채 Adaptor만 학습시켜 파라미터 효율성을 높인다.

$$L_{con} = \mathbb{E}_B \left[ -\log \frac{\exp(\cos(A^I(x_i), A^T(t_i))/\tau)}{\sum_j \exp(\cos(A^I(x_i), A^T(t_j))/\tau)} \right]$$

### 2. Multi-Magnification Descriptive Text Prompts

병리 전문의의 진단 과정(저배율 $\rightarrow$ 고배율)을 모방하여 저배율($l$)과 고배율($h$) 두 가지 해상도에 대한 텍스트 프롬프트를 구성한다. GPT-4와 같은 LLM을 사용하여 각 클래스에 대한 시각적 설명(context)을 생성하고, 여기에 학습 가능한 $M$개의 프롬프트 토큰 $\omega$를 추가하여 다양한 하위 영역의 패턴을 캡처한다.

### 3. Granularity-aware Visual Prompt Learning

시각적 프롬프트 $p^v$를 통해 슬라이드 레벨의 표현을 추출하는 과정은 두 가지 입도로 진행된다.

- **Patch-level Attention**: 모든 개별 패치 특징 $H$를 Key-Value로, 학습 가능한 시각적 프롬프트 $p^v$를 Query로 사용하여 패치 간의 상관관계를 계산한다.
  $$p_{v,p}^{(l)} = \text{Normalize} \left( \text{SoftMax} \left( \frac{p_v K_p^{(l)T}}{\sqrt{d}} \right) V_p^{(l)} \right) + p_v$$
- **Region-level Attention**: 패치들의 좌표를 기반으로 공간 그래프 $G$를 구축하고, Graph Attention Network (GAT)를 통해 인접 패치 간의 메시지 패싱을 수행하여 지역적 구조를 캡처한 super-node 특징을 생성한다. 이후 동일한 어텐션 메커니즘을 적용하여 $p_{v,gr}^{(l)}$를 얻는다.
- **Fusion**: 최종 시각적 특징은 두 입도의 가중 합으로 결정된다: $p_v^{(l)} = (1-\alpha) \cdot p_{v,p}^{(l)} + \alpha \cdot p_{v,gr}^{(l)}$.

### 4. Optimal Transport (OT) for Alignment

시각적 요약 임베딩과 텍스트 프롬프트 임베딩 간의 거리를 측정하기 위해 OT 거리를 사용한다. 시각적 특징 집합 $F$와 텍스트 특징 집합 $G$를 각각 이산 확률 분포 $\mu, \nu$로 정의하고, 두 분포 간의 전송 비용을 최소화하는 최적 운송 계획 $T^*$를 찾는다.

$$d_{OT}(\mu, \nu) = \langle T^*, C \rangle$$

여기서 $C$는 두 특징 간의 거리(Cost matrix)이며, 계산 효율성을 위해 Sinkhorn 알고리즘을 사용하여 엔트로피 정규화된 근사해를 구한다. 최종 예측 확률 $P_c$는 저배율과 고배율의 OT 거리 합의 지수 함수 형태로 계산된다.

## 📊 Results

### 실험 설정

- **데이터셋**: TCGA-NSCLC(폐암), TCGA-RCC(신장암), TCGA-BRCA(유방암) 세 가지 벤치마크를 사용하였다.
- **비교 대상**: Max/Mean pooling, ABMIL, CLAM, TransMIL 등의 MIL 기반 모델과 CoOp, ViLa-MIL, MSCPT 등의 VLM 기반 모델, 그리고 CONCH, QUILT와 같은 Foundation VLM을 비교 대상으로 설정하였다.
- **지표**: AUC, F1-score, Accuracy를 측정하였으며, 16-shot 및 Zero-shot 설정에서 평가하였다.

### 주요 결과

- **Few-shot 성능**: MGPATH(PLIP-G)는 모든 데이터셋에서 기존 MIL 및 VLM 기반 모델들을 압도하였다. 특히 TCGA-BRCA에서 MSCPT(75.82%)와 ViLa-MIL(75.01%)보다 높은 79.56%의 정확도를 달성하였다.
- **Zero-shot 성능**: Foundation VLM인 CONCH 및 QUILT와 비교했을 때, MGPATH(PLIP-G)가 평균적으로 가장 높은 성능을 보였으며, 이는 대규모 사전 학습 데이터와 다중 입도 프롬프트의 결합이 일반화 성능을 크게 향상시켰음을 시사한다.
- **Ablation Study**:
  - **Multi-granular Attention**: 공간적 그룹 어텐션을 제외했을 때 성능이 하락하여, 지역적 구조 캡처의 중요성이 입증되었다.
  - **OT vs Cosine**: 데이터 증강(augmentation) 적용 시 Cosine similarity는 성능이 급격히 하락한 반면, OT는 강건함을 유지하였다.
  - **LLM 영향**: Mistral Medium 3가 생성한 텍스트 프롬프트를 사용했을 때 가장 높은 성능을 보여, 정밀한 시각적 묘사가 성능 향상에 핵심적임을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 대규모 비전 모델(Prov-GigaPath)과 텍스트 모델(PLIP)을 효율적으로 결합하고, 병리 이미지의 특성인 '계층적 구조'를 어텐션 메커니즘에 명시적으로 반영함으로써 Few-shot 분류 성능을 극대화하였다. 특히 단순한 특징 추출을 넘어, GAT를 이용한 공간적 관계 모델링과 OT를 이용한 분포 기반 정렬을 도입한 점이 기술적인 강점이다.

다만, OT 계산이 Cosine similarity보다 계산 비용이 높다는 단점이 있으나, 논문에서 제시한 결과에 따르면 정확도 향상 폭이 계산 비용 증가보다 훨씬 크므로 충분한 트레이드오프가 성립한다. 한계점으로는 매우 거대한 이미지 패치를 처리하기 위한 Flash Attention 등의 최적화 기법이 아직 적용되지 않았으며, 분류 작업을 넘어 세그멘테이션(Segmentation)과 같은 더 정밀한 작업으로의 확장 가능성에 대해서는 논의가 부족하다는 점이 있다.

## 📌 TL;DR

MGPATH는 13억 개의 패치로 학습된 Prov-GigaPath와 PLIP를 결합한 VLM으로, **패치 수준과 지역(Region) 수준의 주의 집중을 동시에 수행하는 다중 입도 프롬프트 학습**과 **노이즈에 강건한 Optimal Transport 기반의 정렬**을 통해 데이터가 극히 적은 Few-shot 상황에서도 최신 SOTA 모델들을 뛰어넘는 WSI 분류 성능을 달성하였다. 이 연구는 대규모 도메인 특화 모델과 정교한 프롬프트 엔지니어링의 결합이 의료 영상 분석의 효율성을 어떻게 높일 수 있는지 보여준다.
