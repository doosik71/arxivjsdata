# Scalable Diffusion Models with Transformers
William Peebles, Saining Xie

## 🧩 Problem to Solve
최근 이미지 생성 모델 분야에서 큰 발전을 이룬 Diffusion Model은 U-Net 아키텍처를 사실상의 표준(de-facto choice) 백본으로 채택해왔습니다. 반면, 자연어 처리(NLP)와 컴퓨터 비전(CV)을 비롯한 대부분의 머신러닝 분야에서는 Transformer 아키텍처가 지배적인 위치를 차지하고 있습니다. 이 연구는 Diffusion Model에서 U-Net의 귀납적 편향(inductive bias)이 성능에 필수적인지 의문을 제기하며, Transformer 아키텍처가 U-Net을 대체하고 Diffusion Model에 Transformer의 우수한 확장성(scalability)과 다른 분야에서의 모범 사례를 적용할 수 있는지 탐구합니다.

## ✨ Key Contributions
*   **Diffusion Transformer (DiT) 도입**: U-Net 백본을 Transformer로 대체한 새로운 Diffusion Model인 DiT를 제안했습니다. 이는 VAE의 잠재 공간(latent space)에서 작동하는 잠재 Diffusion Model에 Transformer를 적용한 것입니다.
*   **확장성 분석**: Gflops로 측정되는 Transformer의 전방 전달(forward pass) 복잡도를 기준으로 DiT의 확장성을 체계적으로 분석했습니다. Transformer의 깊이/너비 증가 또는 입력 토큰 수 증가를 통해 Gflops가 높아질수록 FID(Fréchet Inception Distance)가 일관되게 낮아짐을 발견하여, 모델 Gflops와 샘플 품질 간의 강한 상관관계를 입증했습니다.
*   **U-Net 귀납적 편향의 불필요성 입증**: Diffusion Model 성능에 U-Net의 특정 귀납적 편향이 필수적이지 않음을 보여주며, 표준 Transformer 설계로도 충분히 대체 가능함을 증명했습니다.
*   **최첨단 성능 달성**: 가장 큰 모델인 DiT-XL/2가 class-conditional ImageNet 512×512 및 256×256 벤치마크에서 기존의 모든 Diffusion Model을 능가하며, 256×256 해상도에서 2.27의 FID를 달성하여 최첨단(state-of-the-art) 성능을 기록했습니다.
*   **계산 효율성 입증**: DiT-XL/2가 LDM과 같은 잠재 공간 U-Net 모델보다 계산 효율적이며, ADM과 같은 픽셀 공간 U-Net 모델보다 훨씬 더 효율적임을 보여주었습니다.

## 📎 Related Works
*   **Transformer**: 자연어, 비전, 강화 학습, 메타 학습 등 다양한 도메인에서 도메인 특화 아키텍처를 대체하며 뛰어난 확장성을 보여주었습니다. 픽셀 수준 자동 회귀 모델(Pixel-level autoregressive models) 및 이산 코드북(discrete codebooks)을 사용한 생성 모델에도 활용되었습니다.
*   **Denoising Diffusion Probabilistic Models (DDPMs)**: 이미지 생성에서 GAN을 능가하는 성공을 거두었으며, 개선된 샘플링 기법(예: Classifier-free guidance), 노이즈 예측 방식, 계단식(cascaded) DDPM 파이프라인 등의 발전이 있었습니다. 이 모든 모델에서 U-Net이 기본 백본으로 사용되어 왔습니다.
*   **아키텍처 복잡도**: 이미지 생성 분야에서 아키텍처 복잡도를 측정하는 일반적인 방법은 파라미터 수이지만, 이 논문에서는 이미지 해상도에 따른 영향을 고려하기 위해 Gflops를 주요 척도로 사용합니다. U-Net 아키텍처의 확장성 및 Gflops 특성을 분석한 Nichol 및 Dhariwal의 선행 연구와 관련성이 깊습니다.

## 🛠️ Methodology
본 논문은 Vision Transformer (ViT) [10]의 모범 사례를 따르는 Diffusion Transformer (DiT)를 제안하며, 특히 잠재 Diffusion Model (LDM) [48] 프레임워크 내에서 작동합니다.

1.  **Diffusion Formulation**:
    *   가우시안 Diffusion Model은 실제 데이터 $x_0$에 점진적으로 노이즈를 적용하는 순방향(forward) 노이즈 프로세스 $q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$를 가정합니다.
    *   Diffusion Model은 순방향 프로세스를 역전시키는 역방향(reverse) 프로세스 $p_{\theta}(x_{t-1}|x_t) = \mathcal{N}(\mu_{\theta}(x_t), \Sigma_{\theta}(x_t))$를 학습하도록 훈련됩니다.
    *   간단한 평균 제곱 오차(mean-squared error) 손실 $L_{simple}(\theta) = ||\epsilon_{\theta}(x_t) - \epsilon_t||^2_2$를 사용하여 노이즈 예측 네트워크 $\epsilon_{\theta}$를 훈련합니다.
2.  **Classifier-Free Guidance**:
    *   클래스 라벨 $c$와 같은 추가 정보를 입력으로 받는 조건부 Diffusion Model에서 샘플 품질을 향상시키기 위해 사용됩니다.
    *   $\hat{\epsilon}_{\theta}(x_t, c) = \epsilon_{\theta}(x_t, \emptyset) + s \cdot (\epsilon_{\theta}(x_t, c) - \epsilon_{\theta}(x_t, \emptyset))$ 공식을 통해 가이던스(guidance)를 적용하며, $s > 1$은 가이던스의 강도를 조절합니다.
3.  **Latent Diffusion Models (LDMs)**:
    *   고해상도 픽셀 공간에서 Diffusion Model을 직접 훈련하는 계산 비용 문제를 해결하기 위해 도입되었습니다.
    *   (1) 이미지를 더 작은 공간 표현으로 압축하는 오토인코더(VAE)를 학습하고, (2) 이 잠재 표현 $z=E(x)$의 Diffusion Model을 훈련합니다. DiT는 이 잠재 공간에서 작동합니다.
4.  **Diffusion Transformer (DiT) Design Space**:
    *   **Patchify**: 입력 잠재 표현 $z$ (예: $32 \times 32 \times 4$)를 $T$개의 토큰 시퀀스로 변환합니다. 패치 크기 $p \times p$에 따라 토큰 수 $T=(I/p)^2$가 결정되며, $p$가 작을수록 $T$가 증가하여 Gflops에 큰 영향을 미칩니다. 이 연구에서는 $p=2, 4, 8$을 탐색했습니다.
    *   **DiT Block Design**: 시간 단계 $t$와 클래스 라벨 $c$와 같은 조건부 입력을 처리하는 네 가지 Transformer 블록 변형을 탐색했습니다.
        *   **In-context conditioning**: $t$와 $c$ 임베딩을 추가 토큰으로 입력 시퀀스에 단순히 추가합니다.
        *   **Cross-attention block**: $t$와 $c$ 임베딩을 별도의 시퀀스로 분리하고, 셀프 어텐션 블록 이후에 멀티 헤드 크로스 어텐션 레이어를 추가합니다.
        *   **Adaptive Layer Norm (adaLN) block**: 표준 레이어 정규화(Layer Norm) 레이어를 adaLN으로 대체합니다. 스케일 및 시프트 파라미터 $\gamma, \beta$를 $t$와 $c$ 임베딩의 합으로부터 예측합니다.
        *   **adaLN-Zero block**: adaLN의 변형으로, 잔여 연결(residual connection) 직전에 적용되는 차원별 스케일링 파라미터 $\alpha$를 추가로 예측합니다. $\alpha$를 0 벡터로 초기화하여 각 DiT 블록이 항등 함수(identity function)로 초기화되도록 합니다. 이 방식이 가장 우수한 성능을 보였습니다.
    *   **Model Size**: ViT를 따라 Small (S), Base (B), Large (L), XLarge (XL)의 네 가지 Transformer 설정을 사용하여 모델 크기 $N$(레이어 수), $d$(히든 차원), 어텐션 헤드를 함께 스케일링합니다.
    *   **Transformer Decoder**: 마지막 DiT 블록 후, 이미지 토큰 시퀀스를 원본 공간 입력과 동일한 형상의 노이즈 예측 및 공분산 예측으로 디코딩하기 위해 표준 선형 디코더를 사용합니다.
5.  **Training**: ImageNet 데이터셋에서 256×256 및 512×512 해상도로 Class-conditional latent DiT 모델을 훈련했습니다. AdamW 옵티마이저, $1 \times 10^{-4}$의 고정 학습률, 256의 배치 크기, 수평 뒤집기(horizontal flips) 데이터 증강을 사용했습니다. 가중치에 대한 EMA(Exponential Moving Average)를 유지했습니다.
6.  **Diffusion**: Stable Diffusion [48]에서 사전 훈련된 VAE 모델을 사용했습니다. VAE 인코더는 8배 다운샘플링 팩터를 가집니다. Diffusion 하이퍼파라미터는 ADM [9]의 설정을 따랐습니다.
7.  **Evaluation Metrics**: Fréchet Inception Distance (FID-50K)를 주요 지표로 사용했으며, Inception Score, sFID, Precision/Recall도 보고했습니다. FID 값은 ADM의 TensorFlow 평가 스위트를 사용하여 계산했습니다.

## 📊 Results
*   **DiT Block Design 비교**: adaLN-Zero 블록이 cross-attention 및 in-context conditioning 방식보다 일관되게 낮은 FID를 달성했으며, 가장 계산 효율적이었습니다. adaLN-Zero의 항등 함수 초기화(identity function initialization)는 일반 adaLN보다 성능이 크게 향상되었습니다.
*   **모델 크기 및 패치 크기 확장**: 모델 크기(깊이/너비)를 늘리고 패치 크기(더 많은 입력 토큰)를 줄일수록 FID가 크게 개선됨을 확인했습니다.
*   **Gflops의 중요성**: 모델 파라미터 수는 DiT 모델 품질을 단독으로 결정하지 않았습니다. 파라미터 수를 거의 고정한 채 패치 크기를 줄여 Gflops만 늘려도 성능이 향상되었습니다. 모델 Gflops와 FID-50K 간에 강한 음의 상관관계(-0.93)가 존재하며, 이는 추가적인 모델 계산량이 DiT 모델 성능 향상의 핵심 요소임을 시사합니다.
*   **계산 효율성**: 더 큰 DiT 모델은 총 훈련 계산량 대비 더 효율적이었습니다. 작은 DiT 모델은 더 오래 훈련해도 결국 큰 DiT 모델에 비해 계산 비효율적이 되었습니다.
*   **State-of-the-Art 성능**:
    *   **ImageNet 256×256**: DiT-XL/2 (classifier-free guidance $s=1.50$ 사용)는 FID 2.27을 달성하여 이전 최고 기록인 LDM의 3.60과 StyleGAN-XL의 2.30을 뛰어넘었습니다. 또한 LDM (103.6 Gflops)보다 약간 더 많은 Gflops (118.6 Gflops)로 더 나은 성능을 보여주었고, ADM (1120 Gflops)과 같은 픽셀 공간 U-Net 모델보다 훨씬 효율적이었습니다.
    *   **ImageNet 512×512**: DiT-XL/2 (classifier-free guidance $s=1.50$ 사용)는 FID 3.04를 달성하여 이전 최고 기록인 ADM의 3.85를 능가했습니다. ADM (1983 Gflops)보다 훨씬 적은 Gflops (524.6 Gflops)로 우수한 성능을 보였습니다.
*   **모델 계산량 vs. 샘플링 계산량**: 샘플링 단계를 늘려 샘플링 계산량을 늘리는 것은 모델 계산량 부족을 보완할 수 없었습니다. 예를 들어, DiT-L/2가 1000 샘플링 단계로 XL/2의 128 샘플링 단계보다 5배 많은 계산량을 사용했음에도, XL/2가 더 좋은 FID (23.7 vs 25.9)를 보였습니다.

## 🧠 Insights & Discussion
*   **아키텍처 통합의 가능성**: U-Net 아키텍처의 귀납적 편향이 Diffusion Model 성능에 결정적이지 않다는 발견은 Diffusion Model이 다른 도메인에서 Transformer가 보여준 아키텍처 통합 트렌드에 합류할 수 있음을 의미합니다. 이는 다른 도메인의 모범 사례와 훈련 기법을 상속받고, 확장성, 견고성, 효율성 같은 유리한 특성을 유지할 수 있게 합니다.
*   **확장성의 중요성**: Gflops로 측정되는 모델 계산량의 증가가 Diffusion Model의 샘플 품질 향상에 결정적인 요소임을 보여주었습니다. 이는 파라미터 수뿐만 아니라 실제 계산 복잡도를 고려한 아키텍처 설계의 중요성을 강조합니다.
*   **adaLN-Zero의 효과**: 잔여 블록을 항등 함수로 초기화하는 adaLN-Zero 기법이 Diffusion Transformer의 성능을 크게 향상시켰습니다. 이는 대규모 모델 훈련에서 초기화 전략의 중요성을 다시 한번 확인시켜 줍니다.
*   **향후 연구 방향**: DiT의 유망한 확장성 결과는 더 큰 모델과 토큰 수로 DiT를 확장하는 추가 연구의 필요성을 시사합니다. 또한 DiT를 DALL·E 2 및 Stable Diffusion과 같은 텍스트-이미지 모델의 백본으로 활용하는 연구도 기대됩니다.

## 📌 TL;DR
이 논문은 Diffusion Model의 표준 U-Net 백본을 Transformer로 대체한 **Diffusion Transformer (DiT)**를 제안합니다. 실험 결과, DiT는 Transformer 아키텍처의 뛰어난 확장성을 상속받아, **Gflops가 증가할수록 이미지 생성 품질(FID)이 향상**됨을 입증했습니다. 가장 큰 모델인 **DiT-XL/2는 ImageNet 256×256 및 512×512 벤치마크에서 기존의 모든 Diffusion Model을 능가하며 최첨단 성능을 달성**했습니다. 이는 U-Net의 귀납적 편향이 Diffusion Model에 필수적이지 않으며, Transformer가 이미지 생성 모델의 아키텍처 통합을 이끌 수 있음을 시사합니다.