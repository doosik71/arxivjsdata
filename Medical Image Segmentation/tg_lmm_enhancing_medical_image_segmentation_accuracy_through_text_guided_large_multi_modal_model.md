# TG-LMM: Enhancing Medical Image Segmentation Accuracy through Text-Guided Large Multi-Modal Model

Yihao Zhao, Enhao Zhong, Cuiyun Yuan, Yang Li, Man Zhao, Chunxia Li, Jun Hu, Chenbin Liu (2024)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation)에서 발생하는 정확도 및 일반화 성능의 한계를 해결하고자 한다. 기존의 자동 분할 모델들은 장기의 해부학적 위치와 같은 전문가의 사전 지식(Prior Knowledge)을 효과적으로 활용하지 못하는 문제가 있다. 또한, 기존의 텍스트-시각 모델들은 주로 타겟 객체를 식별(Identification)하는 데 집중할 뿐, 분할의 정밀도(Accuracy) 자체를 높이는 방향으로 설계되지 않았다. 일부 모델이 사전 지식을 활용하려 시도했으나, 대규모 사전 학습 모델(Pre-trained models)을 통합하여 효율성과 성능을 동시에 잡은 사례는 부족했다. 따라서 본 연구의 목표는 전문가의 장기 위치 묘사를 텍스트 가이드로 활용하여 의료 영상 분할의 정밀도를 향상시키는 TG-LMM 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 의료 전문가가 작성한 장기의 공간적 위치 및 관계에 대한 상세한 텍스트 설명을 모델의 프롬프트로 입력하여, 시각적 특징만으로는 부족한 해부학적 맥락을 보완하는 것이다. 이를 위해 사전 학습된 Image Encoder와 Text Encoder를 결합하고, 쿼리 기반의 Feature Mixer를 통해 두 모달리티의 정보를 깊게 융합하여 픽셀 수준의 정밀한 마스크를 생성하는 구조를 제안한다.

## 📎 Related Works

기존의 의료 영상 분할은 U-Net, 3D U-Net, V-Net과 같은 Convolution 기반 방법론이 주를 이루었으나, 이들은 특정 작업에 과적합되는 경향이 있어 새로운 데이터셋에 대한 일반화 성능이 떨어진다. 이후 Transformer 기반의 SAM(Segment Anything Model)과 이를 의료 분야에 적용한 MedSAM이 등장하며 제로샷 학습 및 전이 가능성이 향상되었다. 텍스트 가이드 기반의 분할 연구들도 진행되었으나, 대부분은 텍스트를 통해 타겟을 찾는 수준에 그치거나 분할 마스크 생성 과정에 텍스트 정보를 직접적으로 반영하여 정확도를 높이는 구조가 부족했다. TG-LMM은 단순한 타겟 식별을 넘어, 사전 학습된 거대 모델을 활용해 텍스트 정보를 분할 정밀도 향상의 도구로 사용한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

TG-LMM은 크게 네 가지 구성 요소로 이루어져 있다: **Transformer-based Image Encoder**, **BERT-based Text Encoder**, **Query-based Feature Mixer**, 그리고 **Mask Decoder**이다.

### 주요 구성 요소 및 역할

1. **Image Encoder**: ViT-base 모델을 사용하여 입력 CT 영상을 768차원의 벡터 $F_{im}$으로 추출한다. 입력 영상은 $1024 \times 1024 \times 3$으로 리샘플링된 후 16배 다운샘플링되어 처리된다.
   $$F_{im} = E_{image}(I)$$
2. **Text Encoder**: BERT 기반의 인코더를 통해 전문가의 텍스트 설명을 512차원의 벡터 $F_{text}$로 변환한다. [EOS] 토큰의 활성화 값을 특징 표현으로 사용하며, 이를 선형 투영하여 멀티모달 임베딩 공간으로 보낸다.
   $$F_{text} = E_{text}(T)$$
3. **Feature Mixer**: 쿼리 기반의 융합 모듈로, Self-attention, 두 개의 Cross-attention, 그리고 MLP로 구성된다. 텍스트 벡터가 먼저 Self-attention을 거친 후, 이미지 벡터를 쿼리로 하여 융합된 텍스트 벡터 $F_{fused\_text}$를 생성하고, 다시 이를 쿼리로 하여 융합된 이미지 벡터 $F_{fused\_im}$을 생성한다.
   - $\text{Text Self-attn}: F_{text,1} = F_{text} + \text{attn}(F_{text}, F_{text})$
   - $\text{Text-Image Cross-attn}: F_{text,2} = F_{im} + \text{cross\_attn}(F_{text,1}, F_{im})$
   - $\text{MLP fusion}: F_{fused\_text} = F_{text,2} + \text{MLP}(F_{text,2})$
   - $\text{Image-Text Cross-attn}: F_{fused\_im} = F_{im} + \text{cross\_attn}(F_{fused\_text}, F_{im})$
4. **Mask Decoder**: 융합된 벡터를 픽셀 수준의 마스크로 복원한다. 두 번의 Dilated Convolution을 통해 특징 맵의 크기를 키우고, Bi-directional Transformer를 통해 텍스트 정보와 다시 융합한다. 최종적으로 두 개의 MLP를 통해 세그멘테이션 마스크와 수렴 속도를 높이기 위한 Bounding Box를 동시에 출력한다.

### 학습 절차 및 손실 함수

모델의 총 파라미터는 218M개이며, 오버피팅을 방지하기 위해 Image Encoder와 Text Encoder의 가중치는 동결(Frozen)하고 Feature Mixer와 Mask Decoder(약 5.6M개 파라미터)만 학습시킨다. 손실 함수는 Binary Cross-Entropy(BCE) 손실과 Dice 손실의 단순 합으로 정의된다.

- **BCE Loss**:
$$\mathcal{L}_{BCE}(AGC, GT) = -\frac{1}{P} \sum_{i=1}^{P} [y_i \log a_i + (1-y_i) \log (1-a_i)]$$
- **Dice Loss**:
$$\mathcal{L}_{dice}(AGC, GT) = 1 - \frac{2 \sum y_i a_i}{\sum y_i^2 + \sum a_i^2}$$
- **Total Loss**: $\mathcal{L} = \mathcal{L}_{BCE} + \mathcal{L}_{dice}$

## 📊 Results

### 실험 설정

- **데이터셋**: FLARE, SegTHOR, MSD 데이터셋을 사용하였으며 간, 신장, 췌장, 대동맥 등 다양한 인체 장기를 대상으로 하였다.
- **입력 텍스트**: 권위 있는 의료 교과서 및 가이드라인에서 추출한 장기의 해부학적 위치 및 인접 장기와의 관계에 대한 상세 설명을 사용하였다.
- **평가 지표**: DSC(Dice Similarity Coefficient), $HD_{95}$(95th percentile Hausdorff Distance), ASD(Average Surface Distance)를 사용하였다.
- **비교 대상**: nnUnet, SAM, MedSAM.

### 주요 결과

- **정량적 성과**: DSC 측면에서는 MedSAM과 유사하거나 소폭 높은 성능을 보였으나, 경계선 기반 지표인 $HD_{95}$와 ASD에서 압도적인 성능 향상을 보였다. 특히 전체 데이터셋 평균에서 $HD_{95}$는 29.22 감소하였고, ASD는 9.2 감소하였다.
- **정성적 성과**: 식도(Esophagus)나 하대정맥(Inferior Vena Cava)과 같이 크기가 작고 형태가 연속적인 장기에서 기존 모델 대비 훨씬 정밀한 분할 결과를 나타냈다.
- **Ablation Study**: 텍스트 설명의 복잡도에 따른 실험 결과, 텍스트가 없거나 단순한 이름만 입력했을 때보다 전문가의 상세한 묘사(Complex descriptions)를 입력했을 때 경계선 정밀도($HD_{95}, ASD$)가 비약적으로 향상됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 연구는 단순한 시각 정보에 의존하지 않고, 의료 전문가의 도메인 지식을 텍스트 형태로 주입함으로써 의료 영상 분할의 고질적인 문제인 '경계선 모호함'을 효과적으로 해결하였다. 특히, 파라미터의 대부분을 동결한 상태에서도 높은 성능을 낸 것은 사전 학습된 대규모 모델의 지식을 효율적으로 전이시켰음을 의미하며, 이는 데이터가 부족한 의료 분야에서 매우 실용적인 접근 방식이다.

### 한계 및 비판적 해석

1. **데이터셋 규모의 한계**: 데이터셋이 상대적으로 작아 인코더를 동결해야만 했으며, 이는 의료 도메인에 특화된 최적의 특징 추출이 이루어지지 않았을 가능성을 시사한다.
2. **텍스트 인코더의 효율성**: 일반 자연어-이미지 사전 학습 모델을 사용했기 때문에, 전문 의료 용어에 대한 이해도가 최적화되지 않았을 수 있다.
3. **불완전한 분할**: 사용된 데이터셋들이 특정 장기만 분할하고 나머지는 비워두는 경우가 많아, 인체 전체를 통합적으로 이해하는 모델로 발전시키기에는 데이터의 한계가 있다.

## 📌 TL;DR

TG-LMM은 의료 전문가의 해부학적 위치 묘사를 텍스트 프롬프트로 활용하여 의료 영상 분할의 정확도를 높이는 멀티모달 모델이다. 사전 학습된 ViT와 BERT 인코더를 결합하고 쿼리 기반의 Feature Mixer를 통해 시각-언어 정보를 융합함으로써, 특히 장기의 경계선 묘사($HD_{95}, ASD$)에서 기존 SOTA 모델인 MedSAM보다 뛰어난 정밀도를 달성하였다. 이 연구는 향후 의료 지식 그래프(Knowledge Graph)나 다양한 모달리티의 가이드라인을 통합하여 추가 학습 없이도 새로운 분할 작업을 수행하는 지능형 의료 진단 시스템의 기반이 될 가능성이 높다.
