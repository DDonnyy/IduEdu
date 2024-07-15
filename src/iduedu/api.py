from .facade import IduEduFacade

facade = IduEduFacade()


def get_one():
    return facade.get_one()
