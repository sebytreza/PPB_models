decide <- function(p) {
  p <- sort(p, decreasing = TRUE)
  n <- length(p)
  Fmax <- 0
  kmax <- 0
  F <- 0
  
  C <- matrix(0, nrow = n + 1, ncol = n + 1)
  C[1, 1] <- 1
  
  for (i in 1:n) {
    C[i + 1, 1] <- (1 - p[i]) * C[i, 1]
    for (j in 1:i) {
      C[i + 1, j + 1] <- p[i] * C[i, j] + (1 - p[i]) * C[i, j + 1]
    }
  }
  
  S <- numeric(2 * n)
  for (i in 1:(2 * n)) {
    S[i] <- 1 / i
  }
  
  K <- n
  while (K > 0 && Fmax == F) {
    F <- 0
    for (i in 1:K) {
      F <- F + 2 * i * C[K+1, i+1] * S[i + K]
    }
    for (i in 1:(2 * (K - 1))) {
      S[i] <- p[K] * S[i + 1] + (1 - p[K]) * S[i]
    }
    if (F >= Fmax) {
      Fmax <- F
      kmax <- K
    }
    K <- K - 1
  }
  
  return(list(kmax = kmax, Fmax = Fmax))
}

library(data.table)
library(MLmetrics)

process <- function(x, c, a, SOL, PROBAS) {
  S <- nrow(SOL)
  N <- 11255
  
  F1_top <- 0
  F1_avg <- 0
  F1 <- 0
  
  T <- function(x) {
    x^a / (x^a + (1 - x)^a)
  }
  PRO <- PROBAS / (PROBAS + x * (1 - PROBAS))
  PRO <- c * T(PRO)
  Plim <- numeric(S)

  
  th <- sort(PRO, decreasing = TRUE)[25 * S]
  
  for (i in 1:S) {
    output <- PRO[i, ]
    tar <- SOL[i, ]
    cat(i, "\r")

    sort_idx <- order(output, decreasing = TRUE)
    mask <- which(output > 0)
    decision <- decide(output[mask])
    K <- decision$kmax
    pred <- integer(N)
    pred[sort_idx[1:K]] <- 1
    Plim[i] <- output[sort_idx[K]]
    f1 <- F1_Score(tar, pred, positive = 1)
    if (!is.nan(f1)) {
      F1 <- F1 + f1
    }
    topk <- integer(N)
    topk[sort_idx[1:25]] <- 1
    f1 = F1_Score(tar, topk, positive = 1)
    if (!is.nan(f1)) {
      F1_top <- F1_top + f1
    }
    avgk <- integer(N)
    for (id in 1:N) {
      avgk[id] <- output[id] > th
    }
    if (sum(avgk) == 0) {
      avgk[sort_idx[1]] <- 1
    }
    f1 <- F1_Score(tar, avgk, positive = 1)
    if (!is.nan(f1)) {
      F1_avg <- F1_avg + f1
    }
  }
  cat(F1 / S, F1_top / S, F1_avg / S, "\n")
  return(F1 / S)
  
}

main <- function(x = 10, c = 0.66, a = 0.85) {
  sol <- fread("GLC24_SOLUTION_FILE.csv")$predictions
  pred_file <- fread("predictions_test_dataset_pos.csv", sep = ",")
  
  probas <- as.character(pred_file$probas)
  preds <- as.character(pred_file$predictions)
  
  S <- length(sol)
  N <- 11255
  
  SOL <- matrix(0, nrow = S, ncol = N)
  for (i in 1:S) {
    specs <- as.integer(unlist(strsplit(sol[i], " ")))
    SOL[i, specs + 1] <- 1
  }
  
  PROBAS <- matrix(0, nrow = S, ncol = N)
  for (i in 1:S) {
    r_preds <- as.integer(unlist(strsplit(preds[i], " ")))
    r_probas <- as.numeric(unlist(strsplit(probas[i], " ")))
    PROBAS[i, r_preds + 1] <- r_probas
  }
  
  start <- proc.time()
  process(x, c, a, SOL, PROBAS)
  cat((proc.time() - start)[[3]], "\n")
}

main()