# A Language-Guided Benchmark for Weakly Supervised Open Vocabulary Semantic Segmentation

Prashant Pandey, Mustafa Chasmai, Monish Natarajan, Brejesh Lall

## 🧩 Problem to Solve

점점 더 많은 연구가 Open Vocabulary Semantic Segmentation (OVSS)과 같이 데이터 효율적인 문제 설정으로 집중되고 있습니다. OVSS는 훈련 중에 보지 못했을 수도 있는 임의의 객체를 분할하는 것을 다룹니다. 기존의 Zero-Shot Segmentation (ZSS) 및 Few-Shot Segmentation (FSS) 방법들은 보지 못한 클래스를 분할하기 위해 픽셀 수준으로 레이블링된 학습 데이터(본 클래스)를 사용하지만, 픽셀 수준의 레이블을 얻는 것은 매우 어렵고 비쌉니다. 따라서, 이미지 수준의 레이블과 같은 저렴한 약한 감독(weak supervision)을 사용하여 모델이 학습 데이터에서 보지 못한 새로운 클래스로 일반화할 수 있도록 하는 것이 실제적인 목표입니다. 기존의 프롬프트 기반 학습 방법들은 본 클래스에 과적합되고 계산 비용이 높다는 한계가 있습니다.

## ✨ Key Contributions

* **통합된 약한 감독 OVSS 파이프라인 제안**: 학습 중 픽셀 수준 레이블을 사용하지 않고, 외부 데이터셋이나 파인튜닝 없이 고정된 Vision-Language 모델(CLIP)을 사용하여 Zero-Shot Segmentation (ZSS), Few-Shot Segmentation (FSS) 및 Cross-dataset Segmentation을 수행할 수 있는 Weakly-Supervised Language-Guided Segmentation Network (WLSegNet)를 제안합니다.
* **문제 분리 (Decoupling)**: 약한 ZSS/FSS 문제를 약한 시맨틱 분할(WSS)과 Zero-Shot Segmentation으로 분리하여, WSS, 마스크 제안 생성 및 Vision-Language 모델의 최적화를 개별적으로 용이하게 합니다.
* **새로운 프롬프트 학습 방법**: `mean instance aware prompt learning` (평균 인스턴스 인식 프롬프트 학습)을 제안합니다. 이 방법은 배치 집계(batch aggregates)를 사용하여 이미지 특징에 프롬프트를 매핑함으로써, 프롬프트를 더 일반화 가능하게 하고, 본 클래스에 대한 과적합을 방지하며, 계산 효율성을 높이고, 보지 못한 클래스에도 효율적으로 일반화됩니다.
* **경쟁력 있는 성능 달성**: PASCAL VOC 및 MS COCO 데이터셋에서 기존 약한 감독 기반라인을 큰 폭(예: WZSS에서 39 mIOU 포인트, WFSS에서 3~5 mIOU 포인트)으로 능가하며, 픽셀 기반 강한 감독 방법들과도 경쟁력 있는 결과를 보입니다.
* **Cross-dataset Segmentation 벤치마킹**: MS COCO 이미지 수준 레이블로 훈련하고 PASCAL VOC의 새로운 클래스에서 테스트하여 벤치마킹을 수행합니다.

## 📎 Related Works

* **Zero-Shot Segmentation (ZSS)**:
  * **생성적 방법**: ZS3Net, CagNet 등은 보지 못한 클래스의 합성 특징을 생성합니다.
  * **판별적 방법**: SPNET, LSeg 등은 이미지의 픽셀 수준 특징을 word2vec, fastText와 같은 사전 훈련된 단어 인코더에서 얻은 단어 임베딩에 매핑합니다. STRICT는 의사(pseudo) 레이블 생성기를 사용합니다.
  * **최근 CLIP 기반 방법**: ZegFormer, SimSeg, ZegCLIP 등은 클래스 불가지론적 마스크 제안을 생성하고 사전 훈련된 Vision-Language 모델을 사용하여 영역을 분류합니다.
* **Weakly Supervised Segmentation (WSS)**:
  * 바운딩 박스, 스크리블, 점, 이미지 수준 레이블과 같은 약한 형태의 감독으로 분할 마스크를 생성합니다.
  * **CAM(Class Activation Map) 기반**: EPS, RSCM, CIAN, RCA, L2G 등은 초기 시드 영역을 확장하고 정제합니다. 본 연구에서는 L2G [15]를 의사 레이블 생성에 사용합니다.
* **Semantic Embeddings and Language Models**:
  * word2vec, fastText와 같은 단어 인코더에서 얻은 클래스 이름의 시맨틱 임베딩을 사용합니다.
  * CLIP [21], ALIGN [47]과 같은 트랜스포머 기반 Vision-Language 모델은 대규모 이미지-텍스트 쌍으로 사전 훈련됩니다.
  * **프롬프트 학습**: Zhou et al. [19, 20]은 학습 가능한 프롬프트 컨텍스트 벡터를 사용하여 데이터셋 특정 컨텍스트 정보를 통합했습니다.
* **Zero and Few-Shot Segmentation with Weak Supervision**:
  * 이 분야는 아직 덜 탐구된 영역입니다. [56]은 약한 감독을 위한 메타 학습 접근법을 따릅니다. [57]은 이미지 레이블만을 사용하여 WZSS 설정을 처음 제안했습니다. Open-world segmentation [7]도 관련 연구입니다.

## 🛠️ Methodology

WLSegNet은 약한 이미지 수준 레이블을 사용하여 WZSS, WFSS 및 Cross-dataset Segmentation을 수행하는 통합 파이프라인입니다.

1. **가정 의사 레이블 생성 (Pseudo-label Generation, PLG) 모듈**:
    * L2G [15] 방법을 채택하여 학습 데이터셋 $D_{\text{train}}$의 본 클래스($C_{\text{train}}$)에 대해 다중 타겟 분류 네트워크를 훈련합니다.
    * 입력 이미지 $I$에 대해 CAM(Class Activation Map)으로부터 의사 분할 마스크 $M_{\text{psuedo}}$를 얻습니다. 이 마스크는 다음 단계에서 감독으로 사용됩니다.
    * PLG 모듈은 다른 WSS 방법(예: RCA [17])으로 쉽게 교체 가능하도록 설계되었습니다.

2. **클래스 불가지론적 마스크 생성 (Class-Agnostic Mask Generation, CAMG) 모듈**:
    * MaskFormer [27]를 채택하여 이미지 내의 객체 클래스와 무관하게 클래스 불가지론적 이진 마스크 제안 $M = \{m_1, m_2, \ldots, m_n\}$을 생성합니다.
    * 훈련 시 PLG에서 얻은 의사 레이블 $M_{\text{psuedo}}$만을 MaskFormer의 마스크 손실(Mask Loss)에 대한 감독으로 사용합니다.
    * 각 마스크 제안 $m \in M$에 대해, 입력 이미지 $I$에 $m$을 곱하여 해당 영역 외의 배경을 0으로 만드는 입력 제안 $I_p = \{i_1, i_2, \ldots, i_n\}$을 생성합니다.

3. **CLIP 언어 모델 및 프롬프트 학습**:
    * 사전 훈련된 CLIP 모델(ViT-B/16)은 고정된 상태로 사용됩니다.
    * **Mean Instance Aware Prompt Learning (핵심 기여)**:
        * 데이터셋 컨텍스트 정보를 포착하기 위해 학습 가능한 컨텍스트 벡터 $V = [v]_1 [v]_2 \ldots [v]_k$를 학습합니다.
        * 클래스 $c$에 대한 클래스 프롬프트 제안 $V_c = [v]_1 \ldots [v]_k [w_c]$는 컨텍스트 벡터 $V$와 클래스 임베딩 $w_c$를 연결하여 얻습니다.
        * CLIP Image Encoder에서 얻은 이미지 임베딩 $x$는 얕은 신경망 $h_{\theta}(.)$를 통과하여 인스턴스별 특징 $f$를 얻습니다.
        * 배치 크기 $b$의 입력 배치 $B$에 대해 평균 배치 특징 프로토타입 $\mu_B = \frac{1}{b}\sum f_i$를 계산합니다.
        * 최종 `mean instance aware class prompt` $V_B^c = V_c + \lambda * G_B$를 얻습니다. 여기서 $G_B$는 $\mu_B$가 $(k+1)$번 반복된 행렬이며, $\lambda$는 $\mu_B$가 $V_c$에 추가되는 정도를 조절하는 하이퍼파라미터입니다.
        * $V_B^c$는 사전 훈련된 CLIP 텍스트 인코더 $g(.)$에 입력되어 $t_B^c$ (텍스트 임베딩)를 얻고, 이는 Zero-Shot 분류에 사용됩니다.
        * 클래스 예측 확률은 $p(y=c|x) = \frac{\exp(\text{sim}(x, t_B^c)/\tau)}{\sum_{C_{i=1}} \exp(\text{sim}(x, t_B^i)/\tau)}$로 계산되며, 여기서 $\text{sim}$은 코사인 유사도, $\tau$는 온도 계수입니다.
        * 이 프롬프트 학습 방법은 기존 방법보다 계산 효율적이며 본 클래스에 대한 과적합 위험이 적습니다.

4. **마스크 통합 (Mask Aggregation)**:
    * 각 클래스 $c$에 대해, CLIP 텍스트 인코더에서 얻은 시맨틱 임베딩 $t_B^c$는 CLIP Image Encoder에서 얻은 $M$의 세그먼트 임베딩을 분류하는 가중치로 사용됩니다.
    * 최종 분할 맵 $Z_j(q) = \frac{\sum_i m_{p_i}(q)C_{p_i}(j)}{\sum_k \sum_i m_{p_i}(q)C_{p_i}(k)}$는 겹치는 분류된 제안들을 통합하여 얻습니다. 여기서 $m_{p_i}(q)$는 픽셀 $q$가 $i$번째 마스크 제안 $m_i$에 속할 예측 확률, $C_{p_i}(j)$는 마스크 제안 $m_i$가 $j$번째 범주에 속할 예측 확률입니다.

5. **약한 Zero-Shot 및 Few-Shot 추론**:
    * **WZSS (일반화된 ZSS 포함)**: 각 입력 이미지에 대해 모델은 픽셀을 본 클래스와 보지 못한 클래스로 분할합니다. CLIP이 사용하는 프롬프트는 모든 이미지에 대해 동일하며, 각 클래스에 대한 하나의 프롬프트를 포함합니다.
    * **WFSS**: 특정 쿼리에 대해 예측되는 클래스는 해당 태스크의 서포트 세트의 약한 레이블에 있는 클래스뿐입니다. 따라서 CLIP이 사용하는 프롬프트 세트는 태스크마다 다릅니다. 또한, 배경 클래스의 예측을 정제하기 위해 오프-더-셸프(off-the-shelf) salient detector의 salient map을 활용합니다.

## 📊 Results

* **데이터셋**: PASCAL VOC 2012 (Pascal-5$_{i}$ 분할) 및 MS COCO 2014 (COCO-20$_{i}$ 분할)를 사용합니다.
* **약한 감독 Zero-Shot Segmentation (WZSS)**:
  * PASCAL VOC에서 WLSegNet은 이미지 레이블만을 사용함에도 불구하고, 픽셀 레이블을 사용하는 기존 강한 감독 기반라인 9개 중 6개를 능가합니다 (표 1).
  * 동일한 WZSS 설정을 따르는 DSG [57]보다 본 클래스 mIOU에서 28.8, 보지 못한 클래스에서 37, 조화 평균 IOU에서 39포인트 더 높은 성능을 보입니다.
  * PASCAL VOC 및 MS COCO에서의 ZSS 결과는 강한 감독 기반라인과 비교하여 경쟁력 있는 수준입니다 (표 2, 3).
* **약한 감독 Few-Shot Segmentation (WFSS)**:
  * 1-way 1-shot 설정에서 PASCAL VOC (표 4) 및 MS COCO (표 5)의 모든 약한 감독 기반라인을 각각 최소 7% mIOU, 30% mIOU 이상으로 능가합니다.
  * 더 어려운 2-way 1-shot FSS 설정에서도 PASCAL VOC 및 MS COCO에서 약한 감독 기반라인을 각각 최소 13, 22 mIOU 포인트 이상으로 능가합니다 (표 6, 7).
  * 1-way 5-shot 및 2-way 5-shot 설정에서도 이미지 수준 기반라인을 명확히 능가하며, 픽셀 수준 감독 방법과도 강력한 경쟁자임을 입증합니다 (그림 6-8).
* **Cross-dataset Segmentation**:
  * COCO 데이터셋으로 훈련하고 PASCAL VOC의 새로운 클래스에서 파인튜닝 없이 테스트한 결과 (표 9), 도메인 시프트에도 불구하고 픽셀 기반 방법들과 경쟁력 있는 성능을 달성했습니다.
* **정성적 분석 및 어블레이션 연구**:
  * 다양한 프롬프트 학습 전략과의 비교에서 제안된 `mean instance aware prompt learning`이 가장 높은 조화 평균 mIOU를 달성하며 보지 못한 클래스에서 뛰어난 성능을 보였습니다 (그림 13).
  * 배치 크기에 민감하지 않으며, CAMG 모듈, CLIP 백본, PLG 모듈의 다양한 구성 요소에 대한 어블레이션 연구를 통해 최적의 설계를 확인했습니다 (표 10).
  * t-SNE 플롯(그림 14)을 통해 WLSegNet의 프롬프트가 이미지 및 텍스트 특징의 더 나은 클러스터링과 정렬을 유도하여 학습된 프롬프트의 일반화 가능성을 입증했습니다.

## 🧠 Insights & Discussion

* **실용적인 가치**: WLSegNet은 약한 감독만으로도 높은 성능을 달성함으로써, 고비용의 픽셀 수준 레이블링 없이도 다양한 객체 범주에 걸쳐 일반화할 수 있는 OVSS 모델의 가능성을 보여줍니다. 이는 실제 세계 시나리오에서 데이터 주석 비용을 크게 절감할 수 있는 실질적인 이점을 제공합니다.
* **모델 설계의 강점**: PLG 모듈과 CAMG 및 CLIP 기반 분류 모듈을 분리한 설계는 의사 레이블 생성과 보지 못한 클래스로의 일반화라는 두 가지 핵심 작업을 독립적으로 개발하고 최적화할 수 있도록 합니다. 이는 유연성과 성능 개선 가능성을 높입니다.
* **프롬프트 학습의 중요성**: 제안된 `mean instance aware prompt learning`은 본 클래스에 대한 과적합 문제를 효과적으로 해결하고, 도메인 시프트를 처리하며, 계산 효율성을 유지하면서 보지 못한 클래스에 대한 강력한 일반화 성능을 제공하는 핵심 요소입니다. 이는 Vision-Language 모델의 잠재력을 최대한 활용하는 데 기여합니다.
* **미래 연구 방향**: 본 연구는 약한 감독 OVSS라는 비교적 덜 탐구된 영역에서 강력한 기반라인을 제공하며, 이 분야의 추가적인 연구를 촉진할 것으로 기대됩니다. 특히 약한 감독 설정에서 강력한 일반화 능력을 갖춘 모델 개발에 중요한 통찰을 제공합니다.

## 📌 TL;DR

* **문제**: 기존 시맨틱 분할은 고비용의 픽셀 레이블이 필요하며, 학습 시 보지 못한 새로운 클래스를 분할하는 Open Vocabulary Semantic Segmentation (OVSS)은 약한 감독(예: 이미지 레이블) 환경에서 특히 어렵습니다.
* **해결책**: WLSegNet은 약한 감독 Zero-Shot (WZSS) 및 Few-Shot (WFSS) 분할을 위한 통합 파이프라인으로, 의사 레이블 생성(WSS)과 클래스 불가지론적 마스크 생성을 통해 문제를 분리합니다. 핵심은 배치 평균 특징을 활용하여 본 클래스에 대한 과적합을 피하고 보지 못한 클래스에 잘 일반화되는 새로운 `mean instance aware prompt learning` 방법을 사용하여 고정된 CLIP 모델의 잠재력을 최대한 활용하는 것입니다.
* **결과**: WLSegNet은 PASCAL VOC 및 MS COCO 데이터셋에서 약한 감독 기반라인을 크게 능가하며, 픽셀 수준의 강한 감독 방법과도 경쟁력 있는 성능을 달성합니다. 이는 픽셀 수준 주석 없이도 OVSS를 효과적으로 수행할 수 있음을 보여줍니다.
