# FLanS: A Foundation Model for Free-Form Language-based Segmentation in Medical Images

Longchao Da, Rui Wang, Xiaojian Xu, Parminder Bhatia, Taha Kass-Hout, Hua Wei, Cao Xiao (2025)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation, MIS) 분야에서 기존 모델들이 가진 두 가지 핵심적인 한계를 해결하고자 한다.

첫째, 기존의 foundation model(예: SAM, MedSAM)은 주로 Bounding Box나 Point 기반의 prompt에 의존하며, 텍스트 prompt를 사용하는 경우에도 단순한 클래스 이름(label)만을 입력으로 받는다. 그러나 실제 임상 환경에서 의료진은 "오른쪽 신장을 강조하라" 또는 "가장 큰 장기를 분할하라"와 같이 자연어(Natural Language)를 통해 상호작용하는 경우가 많다. 즉, 자유 형식의 텍스트(Free-form text)를 이해하고 이를 분할 작업으로 연결하는 유연한 모델이 부족한 상황이다.

둘째, 의료 영상의 스캔 방향(Scan Orientation)의 가변성 문제이다. 환자의 자세(supine vs. prone), 촬영 평면, 재구성 알고리즘 등에 따라 동일한 장기가 영상 내에서 서로 다른 위치나 방향으로 나타날 수 있다. 이는 모델이 장기의 해부학적 위치(Anatomical position)와 영상 내의 외관상 위치(Appearance in the scan)를 혼동하게 만들어 분할 정확도를 떨어뜨리는 원인이 된다.

따라서 본 논문의 목표는 자유 형식의 텍스트 prompt를 처리할 수 있는 능력을 갖추고, 동시에 스캔 방향에 관계없이 일관된 결과를 도출하는 의료 영상 분할 foundation model인 **FLanS**를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1.  **RAG 기반의 Free-form Text Prompt Generator**: 임상 전문가의 기록(EMR), 비전문가의 쿼리, 합성 쿼리로 구성된 도메인 말뭉치(Domain corpus)를 활용하여, 실제 의료 현장에서 사용될 법한 다양하고 현실적인 텍스트 prompt를 자동으로 생성하는 RAG(Retrieval Augmented Generation) 프레임워크를 제안한다.
2.  **FLanS 모델 제안**: 해부학적 정보가 포함된 쿼리(Anatomy-Informed)와 해부학적 지식 없이 위치나 크기 기반으로 요청하는 쿼리(Anatomy-Agnostic)를 모두 처리할 수 있는 분할 모델을 구축하였다.
3.  **Symmetry-aware Canonicalization Module**: 입력 영상의 방향성을 표준화된 프레임으로 변환하는 모듈을 통합하여, 영상의 회전이나 반전과 관계없이 일관된 분할 성능을 보장하는 Equivariance/Invariance 특성을 구현하였다.
4.  **대규모 데이터셋 학습 및 검증**: 7개의 공개 데이터셋에서 추출한 10만 장 이상의 의료 영상을 통해 학습하였으며, In-domain 및 Out-of-domain 데이터셋 모두에서 기존 SOTA 모델 대비 우수한 성능을 입증하였다.

## 📎 Related Works

**의료 영상 분할(Medical Image Segmentation)**: 전통적인 방식은 특정 장기에 최적화된 네트워크 설계나 손실 함수 개선에 집중하였으나, 이는 학습 시 사용된 라벨과 정확히 일치하는 쿼리가 필요하다는 제약이 있다. 최근의 SAM 기반 방식들은 Bbox나 Point prompt를 통해 유연성을 높였으나, 텍스트 기반의 묘사적 이해 능력은 여전히 부족하다.

**텍스트 Prompt 분할(Text Prompt Segmentation)**: 자연어 표현을 입력으로 사용하는 Referring Expression Segmentation 연구들이 진행되었으나, 대부분의 의료 분야 연구는 텍스트 주석(annotation) 생성에 과도한 노동력이 소모된다는 한계가 있다. 본 논문은 이를 RAG 기반의 자동 생성 방식으로 해결하여 효율성을 높였다.

**Equivariant Medical Imaging**: 입력 데이터의 변환(회전, 반전 등)에 대해 출력값이 예측 가능하게 변하거나 변하지 않게 하는 Equivariant 네트워크 연구가 존재한다. 본 논문은 Pre-trained 모델을 그대로 활용하면서도 방향성 문제를 해결하기 위해, 아키텍처 자체를 수정하는 대신 입력 데이터를 표준화하는 Canonicalization 방식을 채택하여 차별성을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조
FLanS는 크게 **RAG 기반 쿼리 생성기**, **Text Encoder**, **Canonicalization Network**, 그리고 **Mask Decoder**로 구성된다.

### 1. Retrieval Augmented Query Generator
수동 라벨링의 한계를 극복하기 위해 RAG를 사용하여 두 가지 유형의 쿼리를 생성한다.
*   **Anatomy-Informed Query**: 전문가의 EMR 기록을 Med-BERT로 임베딩하여 저장한 뒤, LLM이 이를 참조하여 "간경변 증상이 보이므로 관련 부위를 분석하라"와 같이 간접적으로 장기를 지칭하는 전문적인 쿼리를 생성한다.
*   **Anatomy-Agnostic Query**: 정답 마스크(GT mask)의 Bbox 정보를 통해 '가장 큰(largest)', '가장 왼쪽의(left-most)' 등 6가지 공간적 카테고리를 정의하고, 이를 RAG를 통해 자연어 문장으로 확장한다.

### 2. Free-Form Language Segmentation
*   **Text Encoding**: CLIP의 Text Encoder를 사용하여 텍스트 prompt $p$를 임베딩 벡터 $t_p \in \mathbb{R}^D$로 변환한다.
*   **Intention Head**: CLIP 임베딩 위에 선형 레이어 $W_{cls} \in \mathbb{R}^{C \times D}$를 추가하여, 텍스트가 어떤 장기를 지칭하는지에 대한 의도(Intention)를 분류한다.
*   **손실 함수**: 마스크 예측을 위한 Dice Loss($L_{Dice}$)와 의도 분류를 위한 Cross-Entropy Loss($L_{ce}$)를 결합하여 학습한다.
$$ \mathcal{L} = \arg \min_{W^*} \frac{1}{|X|} \sum_{x \in X} \frac{1}{|P_x|} \sum_{p \in P_x} \left[ L_{Dice}(\hat{m}_{p,x}, m_{p,x}) + L_{ce}^* \right] $$
여기서 $L_{ce}^* = L_{ce}(\hat{m}_{p,x}, m_{p,x}) + L_{ce}(y_p, \ell_p)$이며, 두 번째 항은 텍스트 임베딩이 정답 장기 클래스와 정렬되도록 유도한다.

### 3. Semantics-Aware Canonicalization
영상 방향의 가변성을 해결하기 위해 $h: X \to \mathcal{G}$라는 Canonicalization 네트워크를 도입한다. 이 네트워크는 입력 영상 $x$를 표준 프레임(Canonical frame)으로 변환하여, 이후의 분할 네트워크 $p$가 항상 일관된 방향의 영상만을 처리하게 한다.
$$ f(x) = \psi_{out}(h(x)) p(\psi_{in}(h^{-1}(x))x, t) $$
이 과정을 통해 모델은 "오른쪽 신장(right kidney)"이라는 텍스트를 받았을 때, 영상 상의 오른쪽이 아닌 해부학적 기준의 오른쪽을 정확히 찾아낼 수 있다.

### 4. 학습 절차
1.  **Canonicalization 학습**: FLARE22 데이터셋에 $O(2)$ 그룹 변환을 적용하고, MSE Loss를 통해 영상을 표준 방향으로 복원하도록 사전 학습한다.
2.  **텍스트 기반 분할 학습**: RAG로 생성된 쿼리를 사용하여 원본 영상에서 분할 성능을 학습한다.
3.  **결합 학습 및 정렬**: 모든 영상에 랜덤한 $O(2)$ 변환을 적용하여 Canonicalization 네트워크와 분할 네트워크를 공동 최적화한다.

## 📊 Results

### 실험 설정
*   **데이터셋**: MSD, BTCV, WORD, AbdomenCT-1K, FLARE22, CHAOS 등 7개 데이터셋에서 24개 장기를 포함한 10만 장 이상의 영상 사용.
*   **평가 지표**: Dice Coefficient 및 Normalized Surface Distance (NSD).
*   **비교 대상**: BiomedParse, UniverSeg, CLIP+MedSAM, MedCLIP+MedSAM 등.

### 주요 결과
*   **Anatomy-Informed Segmentation**: FLanS는 단순 장기 이름뿐만 아니라 자유 형식의 텍스트 쿼리에서도 모든 베이스라인을 압도하는 성능을 보였다. 특히 변환된 데이터셋(TransFLARE 등)에서도 높은 성능을 유지하여 방향 강건성을 입증하였다.
*   **Anatomy-Agnostic Segmentation**: "가장 큰 장기"와 같은 위치/크기 기반 쿼리에서, 정답 Bbox를 직접 입력받은 MedSAM이나 SAM2와 경쟁 가능한 수준의 성능을 달성하였다. 이는 텍스트만으로 공간적 의도를 정확히 파악했음을 의미한다.
*   **Ablation Study**: Canonicalization 모듈을 제거했을 때 변환된 데이터셋에서의 성능이 급격히 하락(Dice 기준 0.895 $\to$ 0.685)하여, 해당 모듈의 필수성이 확인되었다.

## 🧠 Insights & Discussion

본 논문의 핵심 통찰은 단순한 데이터 증강(Data Augmentation)만으로는 의료 영상의 방향성 모호성을 완전히 해결할 수 없다는 점이다. 단순히 회전된 데이터를 많이 학습시키는 것은 모델이 "영상 상의 오른쪽"과 "해부학적 오른쪽"을 혼동하게 만들 수 있다. 반면, **Canonicalization**은 입력 영상을 먼저 표준 해부학적 방향으로 정렬한 뒤 분할을 수행하므로, 텍스트에 포함된 방향성 의미(예: 'right renal')를 물리적 기준에 맞춰 정확히 해석할 수 있게 한다.

또한, t-SNE 시각화를 통해 FLanS의 텍스트 인코더가 서로 다른 장기에 대한 자유 형식의 쿼리들을 의미적으로 잘 군집화하고 있음을 확인하였다. 이는 모델이 단순한 키워드 매칭이 아니라 텍스트의 세부 의미를 이해하고 있음을 시사한다.

**한계점**: 현재 모델은 CT 스캔에 집중되어 있어 MRI나 초음파(Ultrasound) 등 다른 모달리티에서의 성능은 제한적이다. 또한, 현재의 Anatomy-Agnostic 쿼리는 정해진 6가지 카테고리에 국한되어 있어, 향후 "크기 + 위치"와 같은 복합 쿼리를 처리하는 방향으로 확장이 필요하다.

## 📌 TL;DR

FLanS는 의료 영상 분할에서 **자유 형식의 자연어 쿼리**를 처리할 수 있는 Foundation Model이다. **RAG 기반의 쿼리 생성기**를 통해 전문적/일반적 텍스트 데이터를 확보하고, **Symmetry-aware Canonicalization** 모듈을 통해 영상의 방향 가변성 문제를 해결함으로써, 어떤 방향의 스캔 영상에서도 텍스트 지시사항에 따라 정확하게 장기를 분할할 수 있다. 이 연구는 의료진의 자연어 상호작용을 가능하게 하여 임상 진단 보조 및 교육 도구로서의 활용 가능성이 매우 높다.