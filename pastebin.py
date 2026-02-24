
# Added diagnostics plots (appended, existing cells unchanged)
omega = 2.0 * torch.pi

t_eval = torch.linspace(t_min, t_max, 1000, dtype=torch.float64, device=device).view(-1, 1)
t_eval.requires_grad_(True)

model.eval()
x_pred = model(t_eval)
dxdt_pred = grad(x_pred, t_eval)

x_true = x0 * torch.cos(omega * t_eval)
dxdt_true = -x0 * omega * torch.sin(omega * t_eval)

rel_l2_x = float((torch.linalg.norm(x_pred - x_true) / torch.linalg.norm(x_true)).detach().cpu())
rel_l2_v = float((torch.linalg.norm(dxdt_pred - dxdt_true) / torch.linalg.norm(dxdt_true)).detach().cpu())
print(f"Relative L2 error x(t): {rel_l2_x:.4e}")
print(f"Relative L2 error dx/dt: {rel_l2_v:.4e}")

t_np = t_eval.detach().cpu().numpy().reshape(-1)
x_pred_np = x_pred.detach().cpu().numpy().reshape(-1)
x_true_np = x_true.detach().cpu().numpy().reshape(-1)
v_pred_np = dxdt_pred.detach().cpu().numpy().reshape(-1)
v_true_np = dxdt_true.detach().cpu().numpy().reshape(-1)
steps = np.arange(1, len(train_loss_history) + 1)
train_loss_np = np.array(train_loss_history)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

axes[0].plot(t_np, x_pred_np, '-', lw=2, label='PINN')
axes[0].plot(t_np, x_true_np, '--', lw=2, label='Analytic')
axes[0].set_title('Coordinate x(t)')
axes[0].set_xlabel('t')
axes[0].set_ylabel('x')
axes[0].legend()

axes[1].plot(t_np, v_pred_np, '-', lw=2, label='PINN autograd')
axes[1].plot(t_np, v_true_np, '--', lw=2, label='Analytic')
axes[1].set_title('Velocity dx/dt')
axes[1].set_xlabel('t')
axes[1].set_ylabel('dx/dt')
axes[1].legend()

axes[2].plot(steps, train_loss_np, lw=2, color='tab:red')
axes[2].set_yscale('log')
axes[2].set_title('Train loss')
axes[2].set_xlabel('step')
axes[2].set_ylabel('train_loss')

plt.show()
