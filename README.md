# Triple Inverted Pendulum − Swing-Up via TV-LQR

Swing-up and stabilization of a triple inverted pendulum on a cart using trajectory optimization (direct collocation) and time-varying LQR tracking.

### All down → up-up-down (ddd → uud)
![ddd → uud](results/ddd_uud.gif)

### Down-up-down → all up (dud → uuu)
![dud → uuu](results/dud_uuu.gif)

### Down-up-up → up-up-down (duu → uud)
![duu → uud](results/duu_uud.gif)

## Usage

```bash
pip install -r requirements.txt
python run_transition.py
```

Edit `run_transition.py` to change the transition (e.g. `"ddd->uuu"`, `"uud->duu"`).
`u` = up (θ=0), `d` = down (θ=π).

## License

[MIT](LICENSE)
