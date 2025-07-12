# CLIP is Also a Good Teacher: A New Training Framework for Inductive Zero-shot Semantic Segmentation

Jialei Chen, Daisuke Deguchi, Chenkai Zhang, Xu Zheng, and Hiroshi Murase

## 🧩 Problem to Solve

* **일반화된 제로샷 의미론적 분할 (Generalized Zero-shot Semantic Segmentation, GZSS)**은 학습에 사용된 클래스(seen categories)의 supervision만으로 학습 시에는 보지 못했던 클래스(unseen categories)와 보았던 클래스 모두를 분할하는 것을 목표로 합니다.
* 기존 방법들은 CLIP과 같은 대규모 **시각-언어 모델(Vision-Language Models, VLMs)**을 활용하여 제로샷 성능을 달성했지만, VLM은 주로 분류 작업에 설계되었기 때문에 픽셀 단위의 밀집 예측(dense prediction) 작업인 의미론적 분할에 직접 조절 시 최적의 성능을 내기 어렵습니다.
* 기존 VLM 활용 방식들이 겪는 주요 한계점은 다음과 같습니다:
  * **분류 중심 모델의 부적합성**: VLM이 분류에 최적화되어, 의미론적 분할과 같은 하위 작업에 직접 적용 시 성능이 저하됩니다.
  * **복잡한 다단계 학습 과정**: 어댑터(adapter) 기반 또는 마스크 제안자(mask proposer) 기반의 방법들은 복잡하고 시간이 많이 소요되는 다단계 학습 과정을 거쳐야 합니다.
  * **주석 없는 영역 정보의 비활용**: 이미지 내 주석이 없는 영역에 포함된 유용한 시각적-의미론적 정보가 충분히 활용되지 못하여 모델의 잠재적 성능을 제한합니다.

## ✨ Key Contributions

* **CLIP-ZSS (Zero-shot Semantic Segmentation) 프레임워크 제안**: 단순하지만 효과적인 학습 프레임워크를 제안하여 기존 방법들의 한계를 극복합니다.
* **추론 시 VLM 또는 추가 모듈 불필요**: 폐쇄-집합 분할용으로 설계된 **모든** 이미지 인코더(image encoder)가 추론 시 VLM과 결합하거나 새로운 모듈을 추가할 필요 없이 제로샷 및 오픈-어휘(open-vocabulary) 작업을 수행할 수 있도록 지식을 전달합니다.
* **GLM (Global Learning Module) 도입**: CLIP 시각 인코더(CLIP visual encoder)의 `CLS` 토큰에서 이미지 인코더의 밀집 특징(dense features)으로 전역적인 지식을 효과적으로 전달합니다.
* **PLM (Pixel Learning Module) 도입**: 주석 없는 영역에 대해 의미론적으로 구별 가능한 의사 레이블(pseudo labels)을 생성하고, 추가적인 supervision을 위한 의사 프로토타입(pseudo prototypes)을 생성합니다. 이를 위해 다중 스케일 K-평균(multi-scale K-Means)과 마스크 융합(mask fusion)을 활용합니다.
* **최첨단 (SOTA) 성능 달성**: PASCAL VOC, COCO-Stuff, PASCAL Context 세 가지 벤치마크 데이터셋에서 기존 SOTA 방법 대비 hIoU (harmonic mean IoU)에서 각각 2.2%, 1.3%, 8.5%의 큰 성능 향상을 보였습니다.
* **빠른 추론 속도 및 일반화 능력**: CLIP의 도움 없이 오픈-어휘 작업에 적용 가능하며, SOTA 방법보다 최대 7배 빠른 추론 속도를 달성하여 높은 효율성과 일반화 능력을 입증했습니다.

## 📎 Related Works

* **폐쇄-집합 의미론적 분할 (Close-set Semantic Segmentation)**:
  * FCN [41], DeepLab 시리즈 [6, 7]와 같이 픽셀-레벨 분류에 기반한 모델들이 대표적입니다.
  * MaskFormer [13], Mask2Former [12]와 같이 마스크-레벨 분류로 접근하는 모델들도 있습니다.
  * 이 방법들은 사전 정의된 카테고리만 구별 가능하며, 고품질의 대량 주석 데이터가 필수적이라는 한계가 있습니다.
* **제로샷 의미론적 분할 (Zero-shot Semantic Segmentation)**:
  * **VLM 이전 연구**: CLIP [48]과 같은 대규모 VLM이 등장하기 전에는 시각 모델의 특징을 대규모 텍스트로 구성된 의미 공간으로 투영하여 시각과 언어 간의 간극을 줄이려는 연구들이 있었습니다 [24, 56].
  * **VLM 등장 이후 연구**: CLIP, ALIGN [34] 등의 VLM의 뛰어난 제로샷 능력을 하위 작업으로 이전하려는 시도가 활발합니다.
    * **프롬프트(prompt) 또는 어댑터(adapter) 기반**: VLM에 추가적인 학습 가능한 파라미터를 도입하여 밀집 토큰을 하위 작업에 적용합니다 [23, 25, 37, 58, 66, 67].
    * **마스크 제안자(mask proposer) 기반**: 사전 학습된 [49, 59] 또는 온라인 학습 [18, 19, 47] 마스크 제안자를 활용하여 VLM을 객체 수준에서 미세 조정합니다.
  * **기존 VLM 기반 방법들의 공통 한계**: VLM이 분류 작업에 설계되어 분할에 최적화되지 못하고, 학습 과정이 여러 단계로 나뉘며, 주석이 없는 영역의 정보가 낭비된다는 단점이 있습니다.
  * 본 연구는 MaskCLIP [66] 및 ZS3/ZS5 [3]와 관련이 있지만, 추론 시 CLIP에 의존하지 않고, 유도적(inductive) 설정에 적용되며, 학습 후 추가 미세 조정이 필요 없다는 점에서 차별화됩니다.

## 🛠️ Methodology

CLIP-ZSS는 폐쇄-집합 분할을 위해 설계된 **모든** 이미지 인코더가 추론 시 새로운 모듈을 도입하거나 VLM과 결합하지 않고도 제로샷 및 오픈-어휘 작업을 수행할 수 있도록 지식을 전달하는 것을 목표로 합니다 (Fig. 2 참조).

* **학습 설정 (Inductive GZSS)**:
  * 학습 데이터셋의 카테고리는 `seen categories` ($A_s$)와 `unseen categories` ($A_u$)로 엄격히 분리됩니다 ($A_s \cap A_u = \emptyset$).
  * 학습 시에는 $A_s$에 대한 픽셀-레벨 주석만 접근 가능하며, $A_u$에 속하는 주석은 '무시됨(ignored)'으로 처리됩니다.
  * 추론 시에는 $A_s$와 $A_u$ 모두를 분할해야 합니다.

* **전체 구조**: 고정된 CLIP 모델과 학습 가능한 이미지 인코더 외에, 두 가지 핵심 모듈인 Global Learning Module (GLM)과 Pixel Learning Module (PLM)을 도입합니다.

* **Global Learning Module (GLM)**:
  * **목표**: CLIP 시각 인코더의 `CLS` 토큰이 담고 있는 전역적(global) 정보를 이미지 인코더의 밀집 특징(dense features)으로 전달하여 시각과 의미 간의 간극을 줄입니다.
  * **과정**:
        1. 입력 이미지는 학습 가능한 이미지 인코더와 고정된 CLIP 시각 인코더에 각각 입력되어 밀집 특징 $R \in \mathbb{R}^{B \times C \times L}$와 `CLS` 토큰 $S \in \mathbb{R}^{B \times C}$를 얻습니다. (여기서 $B$는 배치 크기, $C$는 채널 수, $L=H \times W$는 밀집 특징의 높이와 너비 곱)
        2. `CLS` 토큰 $S$를 쿼리(Q)로, 밀집 특징 $R$을 키(K)와 값(V)으로 사용하여 어텐션 가중치 $W$를 생성합니다:
            $$ W = \text{Softmax}\left(\frac{S * R}{\sqrt{C}}\right) $$
            여기서 `$*$`는 배치 행렬 곱셈을 의미하며, $W \in [0, 1]^{B \times L}$입니다.
        3. 생성된 가중치 $W$와 밀집 특징 $R$을 곱하여 예측된 `CLS` 토큰 $\hat{S}$를 생성합니다:
            $$ \hat{S} = W * R^T $$
        4. $S$와 $\hat{S}$ 간의 대조 학습(contrastive learning)을 위해 InfoNCE 손실 함수 [43]를 적용합니다:
            $$ L_{\text{global}} = \sum_{i \in B} \frac{\exp(s_i^T \hat{s}_i / \tau)}{\sum_{j \neq i, j \in B} \exp(s_j^T \hat{s}_i / \tau) + \exp(s_i^T \hat{s}_i / \tau)} $$
            여기서 $\tau$는 온도(temperature) 파라미터입니다.
        5. 학습 성능 향상을 위해 이전 `CLS` 토큰들을 저장하는 `CLS` 토큰 뱅크 $V$를 활용하여 더 많은 음성 샘플을 제공합니다.

* **Pixel Learning Module (PLM)**:
  * **목표**: 의미론적 분할을 위한 미세-정밀(fine-grained) 픽셀-레벨 supervision을 제공하며, 특히 주석 없는 영역을 활용합니다 (Fig. 3a 참조).
  * **의사 레이블 생성 (Pseudo label generation)**:
        1. **다중 스케일 K-평균 (Multi-scale K-means)**: CLIP 시각 인코더의 밀집 토큰을 입력으로 사용하여, 다른 크기의 슬라이딩 윈도우로 평균을 내어 마스크 시드(mask seeds) $C_d$를 초기화합니다.
        2. **마스크 융합 알고리즘 (Mask fusion algorithm)**: $C_d$ 간의 코사인 유사도를 계산하여 유사도 임계값 $\lambda$보다 높은 마스크들을 융합합니다. 이는 각 시드와 마스크가 한 번만 사용되도록 합니다.
        3. **결과**: 의미론적으로 구별 가능한 의사 레이블이 생성되며, 이들은 기존 `seen` 레이블과 픽셀 단위로 합쳐져 `fused labels` $F$를 형성하여 추가적인 supervision에 사용됩니다.
  * **합성기 (Synthesizer) / 의사 가중치 생성 (Pseudo weight generation)**:
        1. **목표**: `fused labels` $F$와 이미지 인코더의 밀집 특징을 기반으로 미지의 카테고리에 대한 프로토타입(classifiers)을 생성합니다 (Fig. 3b 참조).
        2. **과정**: $F$에 따라 이미지 인코더의 밀집 특징의 평균을 내어 `seen` 카테고리 및 `잠재적(potential)` 카테고리(의사 레이블 영역에서 추출된)의 중심(centroids) $p_l$을 얻습니다.
        3. 이 중심들을 트랜스포머 디코더(transformer decoder)의 쿼리(Q)로 사용하고, CLIP 시각 인코더의 `CLS` 토큰을 키(K)와 값(V)으로 사용하여 모든 카테고리에 대한 의사 프로토타입을 생성합니다.
        4. 프로토타입은 `seen prototypes` $P_{\text{seen}}$와 `unseen prototypes` $P_{\text{pseudo}}$로 나뉩니다.
        5. $P_{\text{seen}}$은 해당 `seen` 카테고리 이름의 CLIP 텍스트 특징 $T_{\text{seen}}$으로 BCE(Binary Cross-Entropy) 손실을 통해 supervise됩니다:
            $$ L_{\text{bce}} = \sum_{p \in P_{\text{seen}}, t \in T_{\text{seen}}} \log(\text{act}(t^T p)) $$
            여기서 $\text{act}$는 시그모이드 함수입니다.
        6. $P_{\text{pseudo}}$는 `seen`과 `unseen` 카테고리를 추가로 구별하는 분류기로 사용됩니다. 최종 예측 결과 $x$는 다음과 같습니다:
            $$ x = \text{cat}(\alpha R^T T_{\text{seen}}, \beta \cos(R, P_{\text{pseudo}})) $$
            여기서 $\alpha, \beta$는 하이퍼파라미터이며, `cat`은 채널 차원에서의 연결(concatenation)을, $\cos(\cdot, \cdot)$은 코사인 유사도를 나타냅니다.

* **학습 목적 함수**: CLIP-ZSS의 총 학습 목적 함수는 다음과 같습니다:
    $$ L = L_{\text{global}} + L_{\text{nel}}(x, F) + \text{CE}(x, F) + L_{\text{bce}} $$
    여기서 $L_{\text{nel}}$은 NEL loss [67]를, $\text{CE}$는 Cross-Entropy loss를 나타냅니다.

* **추론**:
  * 추론 시에는 학습된 이미지 인코더만 필요하며, 이는 일반적인 의미론적 분할과 동일합니다.
  * 이미지는 이미지 인코더에 입력되어 밀집 특징을 얻습니다.
  * 이 밀집 특징은 CLIP 텍스트 인코더에서 얻은 `seen` 및 `unseen` 카테고리 텍스트 특징과 행렬 곱셈을 수행하여 분류됩니다.
  * 가장 높은 확률을 가진 카테고리가 최종 출력됩니다.

## 📊 Results

* **벤치마크 데이터셋**: PASCAL VOC, COCO-Stuff, PASCAL Context 세 가지 대표적인 벤치마크 데이터셋에서 실험을 수행했습니다.
* **핵심 성능 지표**: `hIoU` (mIoU$_{S}$와 mIoU$_{U}$의 조화 평균), `pAcc` (pixel accuracy), `mIoU` (mean IoU) for seen categories (mIoU$_{S}$) and unseen categories (mIoU$_{U}$)를 사용했습니다.
* **최첨단 (SOTA) 성능 비교**: SegNeXt, Swin Transformer, Segformer 등 다양한 백본(backbone)을 사용하여 기존 SOTA 방법들과 비교한 결과, `CLIP-ZSS`는 모든 데이터셋에서 크게 앞서는 성능을 보였습니다.
  * PASCAL VOC: 기존 SOTA 대비 hIoU 2.2% 향상.
  * COCO-Stuff: 기존 SOTA 대비 hIoU 1.3% 향상.
  * PASCAL Context: 기존 SOTA 대비 hIoU 8.5% 향상.
* **오픈-어휘 일반화 능력**: COCO-Stuff 데이터셋으로 학습하고 PASCAL VOC 및 PASCAL Context의 오픈-어휘 교차-데이터셋 설정에서 테스트했을 때도 뛰어난 성능을 유지했습니다. 특히 ADE20k에서는 일부 부족했지만, 전반적으로 높은 일반화 능력을 보였습니다.
* **추론 효율성**: 추론 시 CLIP 시각 인코더가 필요 없기 때문에, SOTA 방법(DeOP)보다 최대 7배 빠른 속도를 보였으며, 이는 본 방법의 높은 실용성을 시사합니다.
* **정성적 분석**:
  * GLM의 어텐션 맵은 모델이 학습 중에 `seen` 및 `unseen` 카테고리 모두에 대해 객체의 식별 가능한 부분을 정확히 찾아내는 것을 보여주었습니다.
  * PLM의 K-평균 및 마스크 융합 알고리즘은 의미론적으로 유사한 영역을 효과적으로 의사 레이블로 그룹화하여, 심지어 데이터셋에 명시적으로 주석되지 않은 '새로운' 카테고리도 발견할 수 있음을 입증했습니다.
  * 시각화된 예측 결과는 ZegCLIP [67]과 같은 기존 SOTA 방법 대비 `CLIP-ZSS`가 `seen` 및 `unseen` 카테고리 모두에서 더 정확한 분할을 수행함을 보여주었습니다.

## 🧠 Insights & Discussion

* **주요 시사점**: `CLIP-ZSS`는 CLIP의 강력한 지식을 기존의 범용 이미지 인코더에 성공적으로 '전수'하여, 모델의 아키텍처를 변경하거나 추론 시 VLM에 의존하지 않고도 제로샷 및 오픈-어휘 의미론적 분할을 달성할 수 있음을 입증했습니다. 이는 효율적이고 일반화 가능한 분할 시스템을 구축하는 데 중요한 진전입니다.
* **미활용 정보의 혁신적 활용**: 기존 VLM 기반 방법들이 간과했던 주석 없는 영역의 정보를 `PLM`을 통해 의사 레이블 및 의사 프로토타입으로 생성하여 효과적으로 활용함으로써, 모델의 미세-정밀 픽셀-레벨 학습 능력을 크게 향상시켰습니다.
* **성능 향상 메커니즘**: `Global Learning Module`은 CLIP의 `CLS` 토큰에서 전역적인 시각-의미 지식을 추출하여 `unseen` 카테고리에 대한 모델의 판별 능력을 획기적으로 개선했습니다. `Pixel Learning Module`은 주석 없는 영역에 대한 픽셀-레벨 supervision을 제공하여 미세-정밀 분할 성능을 강화했습니다.
* **한계점**:
  * `PLM`의 K-평균 알고리즘은 잔디(grass)와 나무(tree)와 같이 시각적으로 극도로 유사한 카테고리를 완벽하게 구별하는 데 여전히 어려움을 겪을 수 있습니다.
  * 주석 없는 영역에 대한 의사 가중치를 생성하는 합성기(synthesizer)는 현재의 트랜스포머 디코더 기반 설계를 넘어 더욱 정교한 설계 개선이 필요할 수 있습니다.

## 📌 TL;DR

**문제**: 기존 제로샷 의미론적 분할 방법들은 분류 중심의 VLM을 밀집 예측에 부적절하게 적용하고 주석 없는 데이터 영역의 정보를 제대로 활용하지 못하는 한계가 있었습니다.
**제안 방법**: `CLIP-ZSS`는 새로운 학습 프레임워크로, `Global Learning Module (GLM)`을 통해 CLIP `CLS` 토큰의 전역 지식을 표준 이미지 인코더에 전달하고, `Pixel Learning Module (PLM)`을 통해 다중 스케일 K-평균 기반 의사 레이블 생성 및 합성기를 통한 의사 프로토타입 생성을 통해 주석 없는 영역을 효과적으로 활용합니다.
**주요 결과**: `CLIP-ZSS`는 PASCAL VOC, COCO-Stuff, PASCAL Context 등 주요 벤치마크에서 기존 최첨단(SOTA) 방법들을 큰 폭으로 능가하며 hIoU 성능을 향상시켰습니다. 또한, 추론 시 CLIP 모델 없이 작동하여 SOTA 대비 최대 7배 빠른 속도를 보여주며 높은 효율성과 일반화 능력을 입증했습니다.
