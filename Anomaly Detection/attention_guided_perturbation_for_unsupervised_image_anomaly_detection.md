# Attention-Guided Perturbation for Unsupervised Image Anomaly Detection

Yuxuan Cheng, Tingfeng Huang, Yuxuan Cai, Jingbo Xia, Rui Yu, Jinhai Xiang, Xinwei He

## 🧩 Problem to Solve

비지도 이미지 이상 탐지(Unsupervised Image Anomaly Detection, UAD)는 정상 샘플만으로 학습하여 이미지 내의 이상 영역을 자동으로 감지하고 위치를 파악하는 것을 목표로 합니다. 그러나 최신 신경망은 '동일성 지름길(identity shortcut)' 문제에 직면합니다. 이는 재구성 기반 모델이 비정상 샘플까지도 잘 재구성하여 입력과 재구성된 샘플 간의 차이만으로는 이상을 효과적으로 탐지하기 어렵다는 것을 의미합니다. 기존의 섭동(perturbation) 전략들은 입력의 모든 공간적 위치를 동일하게 처리하여, 재구성에 더 중요한 전경(foreground) 위치의 특성을 무시하고 관련 없는 배경에 과적합될 수 있다는 한계가 있습니다.

## ✨ Key Contributions

- **주의-안내 섭동(Attention-Guided Perturbation) 프레임워크 AGPNet 제안**: 기존 섭동 방식과 달리, 각 샘플의 중요한 영역에 표적화된 섭동을 적용하여 재구성 과정을 개선하는 간단하지만 효과적인 재구성 네트워크를 제시합니다.
- **사전 및 학습 가능한 주의 마스크 계산 방식 도입**: 사전 학습된 특징 추출기(frozen feature extractor)에서 얻은 사전 주의(prior attention)와 EMA(Exponential Moving Average) 디코더에서 학습된 주의(learnable attention)를 기반으로 주의 마스크를 계산하는 방법을 제안합니다. 이 마스크는 다양한 산업 이미지에서 정상 패턴을 포괄적으로 학습하도록 재구성 네트워크를 안내합니다.
- **최첨단 성능 달성 및 확장성 입증**: 다중 클래스(multi-class), 단일 클래스(one-class) 설정에서 기존 최첨단 방법보다 우수한 성능을 보였으며, 소수샷(few-shot) 설정에서도 효과적으로 확장되어 뛰어난 이상 탐지 결과를 얻었습니다.

## 📎 Related Works

- **비지도 시각 이상 탐지 (UAD)**:
  - **합성 기반(Synthesizing-based)**: DRAEM, CutPaste와 같이 이상 샘플을 합성하여 훈련에 활용하나, 실제 이상 유형의 다양성을 모두 합성하기 어렵습니다.
  - **임베딩 기반(Embedding-based)**: PaDiM, PatchCore와 같이 사전 학습된 모델로 이미지를 임베딩하고 통계 모델로 정상 분포를 모델링합니다. 많은 계산 자원과 추론 시간이 소요될 수 있습니다.
  - **재구성 기반(Reconstruction-based)**: AutoEncoder, GANs, Diffusion 모델을 사용하며, 정상 샘플은 잘 재구성하고 이상 샘플은 어렵게 재구성한다는 가정을 따릅니다. 하지만 딥러닝 모델의 강력한 일반화 능력으로 인해 '동일성 지름길' 문제가 발생할 수 있습니다. P-Net, AESc, SCADN, DRAEM 등은 섭동을 통해 이 문제를 완화하려 합니다. UniAD, OmniAL, RLR, MambaAD와 같은 최근 연구들은 다중 클래스 이상 탐지 설정을 개선합니다.
- **소수샷 시각 이상 탐지 (Few-shot VAD)**: RegAD, WinCLIP+, PromptAD와 같이 제한된 수의 학습 샘플만으로 이상 탐지를 수행하는 도전적인 설정입니다.

## 🛠️ Methodology

AGPNet은 크게 두 가지 브랜치로 구성됩니다:

1. **메인 재구성 브랜치**:
   - **특징 추출기(Feature Extractor)**: 입력 이미지 $I \in \mathbb{R}^{H \times W \times 3}$를 사전 학습된 DINO(ViT-S-16) 모델 $\phi(\cdot)$에 넣어 특징 맵 $F_l \in \mathbb{R}^{H_{\text{f}} \times W_{\text{f}} \times C_{\text{f}}}$를 추출합니다. 이 특징들은 계층 정규화(layer normalization) 후 합산되어 재구성 대상인 $F_{\text{clean}} = \sum_{l=1}^{L} \sigma(F_l)$이 됩니다. 또한, 특징 추출기에서 얻은 주의 맵 $A_l$은 사전 주의 정보로 활용됩니다.
   - **디코더(Decoder)**: 가벼운 Vision-Transformer 기반 디코더를 사용하여 특징 추출기에서 파생된 주의-안내 섭동(attention-guided perturbed) 표현을 재구성합니다. 이는 '동일성 지름길' 문제를 완화하고 추론 시 가볍고 강력한 성능을 제공합니다.
2. **보조 주의-안내 섭동 브랜치**:
   - **주의 마스크 생성(Attention Mask Generation)**:
     - **사전 주의 마스크 ($A_{\text{prior}}$)**: 특징 추출기의 주의 맵 $\{A_l\}_{l=1}^L$을 집계하여 얻습니다 ($A_{\text{prior}} = \Phi_{\text{aggr}}(\{A_l\}_{l=1}^L)$). 이는 전경 픽셀 및 중요한 위치에 대한 사전 정보를 제공합니다.
     - **학습 가능한 주의 마스크 ($A_{\text{learn}}$)**: 디코더의 EMA(Exponential Moving Average) 버전($\theta_{\text{md}} = \eta \cdot \theta_{\text{md}} + (1-\eta) \cdot \theta_{\text{dec}}$)에서 얻은 self-attention 가중치 $\{A_k\}_{k=1}^K$를 집계하여 얻습니다 ($A_{\text{learn}} = \Phi_{\text{aggr}}(\{A_k\}_{k=1}^K)$). 이는 훈련 초기 단계에서 학습 가능한 주의 가중치의 높은 변동성을 안정화합니다.
     - **최종 주의 마스크 ($A_{\text{final}}$)**: $A_{\text{final}} = \Phi_{\text{norm}}(A_{\text{prior}}) + \Phi_{\text{norm}}(A_{\text{learn}})$으로 두 마스크를 결합합니다 ($\Phi_{\text{norm}}$은 min-max 정규화).
   - **주의-안내 노이즈(Attention-Guided Noise)**:
     - **특징 수준 섭동(Feature-level perturbation)**: $F_{\text{clean}}$에 주의 마스크 $A_{\text{final}}$에 의해 가중된 가우시안 노이즈 $E$를 추가합니다:
       $$F' = F_{\text{clean}} + E \odot (\alpha(t) \cdot \Phi_{\text{norm}}(A_{\text{final}}) + \beta)$$
       여기서 $\alpha(t)$는 훈련 epoch에 따라 노이즈 강도를 선형적으로 증가시킵니다.
     - **이미지 수준 섭동(Image-level perturbation)**: $A_{\text{final}}$을 이미지 크기로 업샘플링하여 얻은 $A_{\text{img}}$를 바탕으로 마스크 영역에 가우시안 노이즈를 적용합니다. 마스크 비율은 점진적으로 증가합니다.
3. **손실 함수(Loss Function)**:
   - 전체 손실은 특징 섭동에 의한 재구성 손실 $L_{\text{feat}}$와 이미지 및 특징 섭동에 의한 재구성 손실 $L_{\text{img,feat}}$의 평균으로 계산됩니다:
     $$L_{\text{total}} = \frac{1}{2}(L_{\text{feat}} + L_{\text{img,feat}})$$
   - 각 손실은 MSE(Mean Squared Error)를 사용하여 계산됩니다:
     $$\begin{cases} L_{\text{feat}} = \frac{1}{H_{\text{f}} \times W_{\text{f}}} \text{MSE}(F_{\text{feat}}, F_{\text{clean}}) \\ L_{\text{img,feat}} = \frac{1}{H_{\text{f}} \times W_{\text{f}}} \text{MSE}(F_{\text{img,feat}}, F_{\text{clean}}) \end{cases}$$
4. **이상 맵(Anomaly Map)**:
   - 추론 시, 재구성 네트워크의 입력 특징 $F_q$와 출력 특징 $\hat{F}_q$ 사이의 픽셀 단위 재구성 오류 $M_{h,w} = \|F_{h,w} - \hat{F}_{h,w}\|_2$를 계산합니다. 이 오류 맵은 원본 이미지 크기로 업샘플링되어 최종 이상 맵이 되며, 이미지 수준 이상 점수는 이상 맵의 최대값을 통해 얻습니다.

## 📊 Results

- **MVTec-AD 벤치마크**:
  - **다중 클래스 설정**: I-AUC 98.7%, P-AUC 98.0%, PRO 92.9%로 UniAD, DiAD, ViTAD, MambaAD를 포함한 기존 최첨단 방법들을 능가하며 최고의 성능을 달성했습니다.
  - **단일 클래스 설정**: I-AUC 99.2%, P-AUC 98.3%로 DRAEM, DeSTSeg보다 우수하며, SimpleNet과 비교하여 강력한 일반화 성능을 보여주었습니다.
- **VisA 벤치마크**: I-AUC 92.3%, P-AUC 98.4%로 UniAD, DiAD, ViTAD 대비 현저히 우수한 성능을 보였습니다.
- **MVTec-3D 벤치마크**: I-AUC 84.9%, P-AUC 98.0%, PRO 92.9%로 모든 평가 지표에서 UniAD, DiAD, ViTAD를 앞섰습니다.
- **소수샷 이상 탐지(MVTec-AD)**: 2-shot에서 P-AUC 96.2%, 4-shot에서 P-AUC 97.3%로 PromptAD, PatchCore를 포함한 소수샷 전용 모델들 사이에서 선도적인 성능을 보였습니다. 특히 픽셀 수준 이상 위치 파악에서 강점을 나타냈습니다.
- **효율성 비교**: Learnable Parameters 및 FPS 측면에서 두 번째로 높은 효율성을 보였으며, FLOPs는 가장 높았지만 전반적으로 효율성과 성능의 균형이 좋음을 입증했습니다.
- **어블레이션 스터디**:
  - **주의-안내 노이즈**: 무작위 노이즈나 노이즈를 추가하지 않는 경우보다 주의-안내 노이즈(특히 이미지 및 특징 수준 모두에서 적용된 하이브리드 방식)가 이상 탐지 및 위치 파악 성능을 크게 향상시킴을 확인했습니다.
  - **백본의 일반성**: WideResNet-50, EfficientNet-b4, CLIP(ViT-B), DINO(ViT-S) 등 다양한 CNN 및 ViT 기반 백본에 대해 일관되게 경쟁력 있는 성능을 보여 방법론의 일반성을 입증했습니다.
  - **다층 특징(Multi-Layer Features)**: 얕은(shallow) 특징과 깊은(deep) 특징을 모두 통합할 때 최적의 성능을 달성하며, 이는 다양한 세분화된 이상 분석에 상호작용이 중요함을 시사합니다.
  - **평균 교사 디코더(Mean Teacher Decoder)**: 평균 교사 디코더를 사용하면 학습 가능한 주의 가중치의 높은 변동성을 완화하여 탐지 및 위치 파악 성능이 향상됨을 확인했습니다.
  - **노이즈 강도 $\alpha$**: '쉬움-어려움(easy-to-hard)' 전략으로 노이즈 강도 $\alpha$를 선형적으로 증가시킬 때 가장 좋은 성능을 보였으며, 이는 모델이 점진적으로 학습할 수 있도록 돕는 효과적인 방법임을 입증했습니다.
  - **주의 맵 구성 요소**: 사전 주의 ($A_{\text{prior}}$)와 학습 가능한 주의 ($A_{\text{learn}}$)를 모두 결합했을 때 최고의 성능을 달성했습니다.

## 🧠 Insights & Discussion

AGPNet은 주의-안내 섭동을 통해 재구성 네트워크가 이상을 잘 재구성하는 '동일성 지름길' 문제를 효과적으로 해결합니다. 샘플별 주의 마스크는 이미지 내 중요한 전경 영역에 섭동을 집중시켜, 모델이 정상 패턴을 더 명확하고 포괄적으로 학습하도록 강제합니다. 이 전략은 다양한 범주와 이상 유형에 유연하게 대응하며, 특히 데이터가 제한적인 소수샷 환경에서 모델의 수렴 속도를 높이고 사용 가능한 학습 샘플의 활용도를 극대화합니다. 추론 시에는 보조 브랜치가 제거되어 추가적인 계산 비용 없이 효율적으로 작동합니다.

제한점으로는, 제안된 방법이 훈련 세트에 노이즈가 포함된 샘플에 민감할 수 있다는 점이 있습니다. 향후 연구에서는 이러한 프레임워크의 강건성을 높이는 데 초점을 맞출 가치가 있습니다. 또한, 이 연구는 주로 산업 이상 탐지에서 우수한 성능을 입증했지만, 의료 영상이나 비디오 이상 분석과 같은 다른 분야에서의 일반성은 더 광범위한 조사가 필요합니다.

## 📌 TL;DR

- **문제**: 재구성 기반 비지도 이미지 이상 탐지 모델은 '동일성 지름길(identity shortcut)' 문제로 인해 정상 및 비정상 샘플을 모두 잘 재구성하여 이상 탐지가 어렵고, 기존 섭동 방식은 중요한 영역을 간과합니다.
- **해결책**: AGPNet은 재구성 브랜치와 보조 주의-안내 섭동 브랜치로 구성됩니다. 보조 브랜치는 사전 학습된 특징 추출기에서 얻은 '사전 주의'와 EMA 디코더에서 학습된 '학습 가능한 주의'를 결합하여 최종 주의 마스크를 생성합니다. 이 마스크는 이미지 및 특징 수준에서 가우시안 노이즈 섭동을 안내하여 모델이 중요한 정상 패턴을 보다 효율적이고 포괄적으로 학습하도록 강제합니다.
- **결과**: AGPNet은 MVTec-AD, VisA, MVTec-3D 벤치마크에서 다중 클래스, 단일 클래스, 소수샷 설정 모두에서 최첨단 성능을 달성했습니다. 특히 이상 위치 파악 및 효율성 면에서 뛰어났으며, 제안된 주의-안내 섭동 전략의 효과를 강력하게 입증했습니다.
