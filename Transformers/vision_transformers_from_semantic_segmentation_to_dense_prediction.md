# Vision Transformers: From Semantic Segmentation to Dense Prediction

Li Zhang, Jiachen Lu, Sixiao Zheng, Xinxuan Zhao, Xiatian Zhu, Yanwei Fu, Tao Xiang, Jianfeng Feng, Philip H.S. Torr

## 🧩 Problem to Solve

기존의 합성곱 신경망(CNN) 기반 완전 합성곱 네트워크(FCN)는 점진적인 다운샘플링과 제한된 지역 수용장(receptive field)으로 인해 이미지의 장거리 의존성(long-range dependency)을 효과적으로 모델링하는 데 한계가 있습니다. 이는 의미론적 분할(semantic segmentation)과 같은 Dense Prediction 태스크에서 중요한 제약으로 작용합니다. 반면, 초기 Vision Transformer(ViT) 모델은 이미지 분류에서는 성공적이었지만, 피라미드 구조의 부재, 지역 문맥 처리 능력 부족, 그리고 상당한 계산 비용으로 인해 객체 검출 및 인스턴스 분할과 같은 광범위한 Dense Prediction 애플리케이션에는 덜 효과적이었습니다. 이 논문은 이러한 한계들을 극복하고 Transformer를 Dense Prediction에 더 효과적으로 적용하는 것을 목표로 합니다.

## ✨ Key Contributions

- **SETR (SEgmentation TRansformer) 제안:** 컴퓨터 비전 분야에서 Transformer를 Dense Prediction 태스크(특히 의미론적 분할)에 처음으로 적용하여, 기존 FCN 설계에 대한 새로운 대안을 제시했습니다.
- **전역 문맥 모델링:** 이미지 패치를 시퀀스로 취급하고, Transformer 인코더를 통해 각 레이어에서 전역 문맥 정보를 학습함으로써 FCN의 제한된 수용장 문제를 해결했습니다.
- **다양한 디코더 설계:** SETR의 인코더로부터 학습된 특징 표현의 효과를 검증하기 위해 Naive, Progressive Upsampling (PUP), Multi-Level feature Aggregation (MLA) 세 가지 디코더 설계를 제시했습니다.
- **HLG (Hierarchical Local-Global) Transformer 아키텍처 도입:** 기본적인 ViT의 한계를 극복하기 위해 계층적인 구조와 지역(local) 및 전역(global) 어텐션 메커니즘을 결합한 새로운 Transformer 백본을 개발했습니다. 이는 비용 효율성을 높이고 다양한 Dense Prediction 태스크에 적용 가능합니다.
- **최첨단 성능 달성:** 의미론적 분할(ADE20K, Cityscapes), 이미지 분류(ImageNet-1K), 객체 검출 및 인스턴스 분할(COCO) 등 여러 비전 태스크에서 기존 모델들보다 우수하거나 경쟁력 있는 성능을 입증했습니다.

## 📎 Related Works

- **의미론적 분할:**
  - **FCN 기반:** FCN (Long et al., 2015)의 등장 이후 CRF/MRF (Chen et al., 2015)를 통한 개선, Encoder-Decoder 구조 (Badrinarayanan et al., 2017; Ronneberger et al., 2015).
  - **수용장 확대 및 문맥 모델링:** Dilated convolution (DeepLab, Yu & Koltun, 2016), Pyramid Pooling Module (PSPNet, Zhao et al., 2017), ASPP (DeepLabV2, Chen et al., 2017).
  - **어텐션 기반:** PSANet (Zhao et al., 2018), DANet (Fu et al., 2019), CCNet (Huang et al., 2019), DGMN (Zhang et al., 2020) 등 장거리 문맥을 포착하는 방법들.
- **Vision Transformer:**
  - **NLP 기원:** Transformer (Vaswani et al., 2017)의 자연어 처리 성공.
  - **초기 이미지 적용:** Non-local network (Wang et al., 2018), AANet (Bello et al., 2019), Axial-Attention (Wang et al., 2020).
  - **순수 Transformer:** ViT (Dosovitskiy et al., 2021)의 이미지 분류에서의 성공.
  - **계층적 및 효율적 ViT:** PVT (Wang et al., 2021), Swin Transformer (Liu et al., 2021b) 등 계층적 특징과 효율적인 어텐션을 도입한 후속 연구들. 본 연구의 SETR은 Dense Prediction에 Transformer를 적용한 최초 시도 중 하나이며, HLG는 이들 연구의 동시적인 발전에 기여합니다.

## 🛠️ Methodology

- **SETR (SEgmentation TRansformer) (단일 단계 Transformer):**
  - **이미지 패치화:** 입력 이미지 $x \in R^{H \times W \times 3}$를 $16 \times 16$ 크기의 고정된 패치 그리드로 분해합니다. 각 패치는 선형 임베딩을 통해 잠재적인 $C$-차원 임베딩 벡터로 변환되고, 위치 임베딩($p_i$)이 추가되어 1D 시퀀스 $E = \{e_1+p_1, \dots, e_L+p_L\}$가 생성됩니다. 여기서 $L = (H/16) \times (W/16)$.
  - **Transformer 인코더:** $L_e$개의 Multi-Head Self-Attention (MSA) 및 Multilayer Perceptron (MLP) 블록으로 구성된 표준 Transformer 인코더가 $E$를 입력으로 받아 특징을 학습합니다. 각 Transformer 레이어는 전역 수용장을 가집니다.
    - Self-Attention (SA) 계산: $\text{SA}(Z^{l-1}) = Z^{l-1} + \text{softmax}\left(\frac{Z^{l-1}W_Q(Z^{l-1}W_K)^\top}{\sqrt{d}}\right)(Z^{l-1}W_V)$
  - **디코더 설계:**
    - **Naive:** 인코더 마지막 특징 $Z^{L_e}$를 범주 수 차원으로 투영한 후, 원본 해상도로 선형 보간(bilinear upsampling).
    - **PUP (Progressive Upsampling):** $2 \times$ 업샘플링과 합성곱 레이어를 4번 반복하여 점진적으로 해상도를 복구합니다.
    - **MLA (Multi-Level feature Aggregation):** 인코더의 여러 Transformer 레이어($Z_6, Z_{12}, Z_{18}, Z_{24}$)에서 추출된 특징들을 집계하여 다중 수준 특징 융합을 수행합니다.
- **HLG (Hierarchical Local-Global) Transformer (다목적 백본):**
  - **피라미드 구조:** 이미지의 특징 표현을 다양한 스케일($H/4 \times W/4$부터 $H/32 \times W/32$)로 학습하는 4단계 구조를 가집니다.
  - **HLG Transformer 레이어:**
    - **로컬 어텐션:** 고정된 윈도우($R \times R$) 내에서 어텐션을 적용합니다.
      - **일반 로컬 어텐션:** 윈도우 내 인접한 패치 간의 정보 교환에 중점을 둡니다.
      - **Dilated 로컬 어텐션:** 간격을 두고 패치를 샘플링하여 효율적으로 장거리 의존성을 학습합니다.
    - **전역 어텐션:** 각 윈도우에서 1차원 특징 벡터를 추출하는 윈도우 임베딩(예: 평균 풀링)을 통해 압축된 전역 특징 맵 $Z_G$를 생성합니다. 로컬 특징 $Z_L$이 $Z_G$에 대해 쿼리하는 방식으로 효율적인 전역 문맥 학습을 수행합니다.
    - **파라미터 및 계산 공유:** 로컬 및 전역 어텐션의 $W_Q, W_K, W_V$ 파라미터를 공유하고, 계산을 최적화합니다.
    - **DWMLP (Depth-Wise MLP):** 효율성을 위해 depth-wise convolution과 squeeze-and-excitation을 포함하는 MLP를 사용하며, MLP의 스트라이드 설정을 통해 다운샘플링을 수행합니다.
    - **계층적 로컬-전역 어텐션:** 일반 로컬 어텐션과 dilated 로컬 어텐션을 순차적으로 엮어 하나의 HLG Transformer 레이어를 구성합니다.
  - **HLG 기반 디코더:** 의미론적 분할을 위해 HLG의 모든 4단계 특징을 $H/16 \times W/16$ 크기로 보간하고, SETR-PUP과 유사한 방식으로 최종 분할 결과를 얻습니다. 객체 검출에는 RetinaNet, 인스턴스 분할에는 Mask R-CNN과 같은 기존 디코더 프레임워크를 사용합니다.

## 📊 Results

- **의미론적 분할 (SETR):**
  - **ADE20K:** SETR-MLA는 단일 스케일(SS) 추론에서 48.64% mIoU, 다중 스케일(MS) 추론에서 50.3% mIoU를 달성하여 새로운 최첨단 기록을 세웠습니다. 특히 BEiT 사전 학습 시 53.1% mIoU로 성능이 크게 향상되었습니다.
  - **Cityscapes:** SETR-PUP은 82.2% mIoU를 달성하여 FCN 기반 및 어텐션 기반 접근 방식보다 우수하거나 경쟁력 있는 성능을 보였습니다.
  - 사전 학습의 중요성: ImageNet-21K 및 BEiT 사전 학습이 성능에 결정적인 영향을 미쳤습니다.
  - 보조 손실(Auxiliary loss)은 모델 학습, 특히 초반 최적화에 큰 도움이 되는 것으로 나타났습니다.
  - ViT는 FCN에 비해 높은 FLOPS와 GPU 메모리를 사용하지만, 더 큰 사전 학습 데이터셋과 기법을 활용할 때 우수한 성능을 보여주며, 향후 계산 비용 최적화가 기대됩니다.
- **이미지 분류 (HLG Transformer on ImageNet-1K):**
  - HLG Transformer는 DeiT, PVT, Swin 등 다른 ViT 모델들과 비교하여 모든 규모에서 Top-1 정확도 및 효율성 측면에서 뛰어난 성능을 보였습니다. 예를 들어, HLG-Tiny는 PVT-T보다 6%P 높은 81.1%를 달성했고, HLG-Large는 Swin-B보다 더 적은 파라미터로 84.1%를 달성했습니다.
- **객체 검출 및 인스턴스 분할 (HLG Transformer on COCO):**
  - **RetinaNet (객체 검출):** HLG Transformer는 ResNet, ResNeXt와 같은 CNN 백본 및 PVT, Swin, RegionViT 등 다른 ViT 백본을 일관되게 능가했습니다. HLG-Tiny는 PVTv2-B1보다 1.3%P 높은 AP를 달성했습니다.
  - **Mask R-CNN (인스턴스 분할):** HLG Transformer는 유사하게 다른 백본들보다 일관되게 뛰어난 성능을 보였습니다. HLG-Tiny는 ResNet-18보다 7.8%P, PVT-T보다 4.9%P 높은 AP를 기록했습니다.
- **정성적 결과:** HLG Transformer는 PVT와 비교하여 작은 객체를 더 정확하게 검출하고, 보행자, 난간, 교통 표지판 등 미세한 객체 인스턴스를 더 잘 분할하는 능력을 보여주었습니다.
- **HLG에 대한 어블레이션 연구:** 계층적 로컬 어텐션(일반 + dilated) 적용 시 성능이 0.64%P 향상되었고, DWMLP와 로컬/전역 어텐션 공유 등 핵심 구성 요소들이 점진적인 성능 향상을 가져왔음을 확인했습니다.

## 🧠 Insights & Discussion

- **Dense Prediction을 위한 Transformer의 가능성:** Transformer는 전역적인 문맥 모델링 능력을 통해 기존 CNN 기반 FCN의 고질적인 문제인 제한된 수용장 한계를 효과적으로 해결하며, 의미론적 분할 등 Dense Prediction 태스크에서 강력한 대안이 될 수 있음을 입증했습니다.
- **ViT의 실용성 강화:** HLG Transformer는 바닐라 ViT의 높은 계산 비용과 피라미드 구조 부재 등의 단점을 보완하여, ViT를 단순 분류를 넘어 객체 검출, 인스턴스 분할 등 다양한 Dense Prediction 태스크에 적용할 수 있는 범용적이고 비용 효율적인 백본으로 만들었습니다.
- **계층적 구조와 하이브리드 어텐션의 중요성:** HLG 아키텍처의 성공은 계층적 특징 처리와 더불어, 효율적인 로컬 어텐션(특히 dilated 버전)과 전역 어텐션의 시너지를 통해 지역적 세부 정보와 전역적 문맥 정보를 모두 효과적으로 포착할 수 있었기 때문입니다.
- **사전 학습의 결정적 역할:** Transformer 모델의 성능은 대규모 데이터셋(ImageNet-21K) 또는 BEiT와 같은 고급 사전 학습 기법에 크게 의존한다는 점을 강조합니다. 이는 Transformer의 데이터 효율성 측면에서의 과제를 시사하기도 합니다.
- **계산 비용과 성능의 균형:** Transformer는 CNN보다 계산 비용이 높지만, HLG와 같은 설계를 통해 합리적인 수준에서 뛰어난 성능을 달성할 수 있음을 보여줍니다. 향후 하드웨어 발전 및 최적화 연구를 통해 이러한 비용은 더욱 줄어들 것으로 예상됩니다.

## 📌 TL;DR

기존 CNN의 제한된 수용장 문제를 극복하고 Transformer를 Dense Prediction 태스크에 도입하고자, 본 논문은 이미지 패치 시퀀스 학습 기반의 **SETR(SEgmentation TRansformer)**과 비용 효율적인 계층적 아키텍처인 **HLG(Hierarchical Local-Global) Transformer**를 제안합니다. SETR은 의미론적 분할에서 최첨단 성능을 달성했으며, HLG는 지역/전역 어텐션 메커니즘과 피라미드 구조를 통해 이미지 분류, 객체 검출, 인스턴스 분할 등 다양한 Dense Prediction 태스크에서 우수한 성능을 입증하여, Transformer가 범용적이고 강력한 비전 백본이 될 수 있음을 보여주었습니다.
