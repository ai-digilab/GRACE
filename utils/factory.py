def get_model(model_name, args):
    name = model_name.lower()
    if name == "grace":
        from models.grace import GRACE

        return GRACE(args)
    else:
        assert 0
