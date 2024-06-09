from supertokens_python.recipe import session, thirdparty, emailpassword, dashboard, userroles
from supertokens_python.recipe.thirdparty.provider import ProviderInput, ProviderConfig, ProviderClientConfig
from supertokens_python import (
    InputAppInfo,
    SupertokensConfig,
)
from supertokens_python import InputAppInfo
from supertokens_python.recipe.emailpassword.interfaces import RecipeInterface, SignUpOkResult
from typing import Dict, Any
from supertokens_python.recipe.userroles.asyncio import add_role_to_user
from supertokens_python.recipe.userroles.interfaces import UnknownRoleError
from supertokens_python.recipe.userroles.asyncio import create_new_role_or_add_permissions
from supertokens_python.recipe.thirdparty.interfaces import RecipeInterface as ThirdPartyRecipeInterface
from supertokens_python.recipe.thirdparty.types import RawUserInfoFromProvider

# To create a new role
async def create_role():
    res = await create_new_role_or_add_permissions("admin", ["read", "delete", "edit"])
    if not res.created_new_role:
        # The role already existed
        pass

# Function to add a role to a user
async def add_role_to_user_func(user_id: str, role: str):
    res = await add_role_to_user("public", user_id, role)
    if isinstance(res, UnknownRoleError):
        # No such role exists
        return

    if res.did_user_already_have_role:
        # User already had this role
        pass

def override_emailpassword_functions(original_implementation: RecipeInterface) -> RecipeInterface:
    original_sign_up = original_implementation.sign_up
    
    async def sign_up(
        email: str,
        password: str,
        tenant_id: str,
        user_context: Dict[str, Any]
    ):
        result = await original_sign_up(email, password, tenant_id, user_context)

        if isinstance(result, SignUpOkResult):
            id = result.user.user_id
            email = result.user.email
            print(id)
            print(email)
            await add_role_to_user_func(id, "user")
        
        return result
                
    original_implementation.sign_up = sign_up

    return original_implementation

def override_thirdparty_functions(original_implementation: ThirdPartyRecipeInterface) -> ThirdPartyRecipeInterface:
    original_thirdparty_sign_in_up = original_implementation.sign_in_up

    async def thirdparty_sign_in_up(
        third_party_id: str,
        third_party_user_id: str,
        email: str,
        oauth_tokens: Dict[str, Any],
        raw_user_info_from_provider: RawUserInfoFromProvider,
        tenant_id: str,
        user_context: Dict[str, Any]
    ):
        result = await original_thirdparty_sign_in_up(third_party_id, third_party_user_id, email, oauth_tokens, raw_user_info_from_provider, tenant_id, user_context)

        # user object contains the ID and email of the user
        user = result.user
        print(user)

        # This is the response from the OAuth 2 provider that contains their tokens or user info.
        provider_access_token = result.oauth_tokens["access_token"]
        print(provider_access_token)

        # if result.raw_user_info_from_provider.from_user_info_api is not None:
            # first_name = result.raw_user_info_from_provider.from_user_info_api["first_name"]
            # print(first_name)

        if result.created_new_user:
            print("New user was created")
            id = result.user.user_id
            await add_role_to_user_func(id, "user")

            # TODO: Post sign up logic
        else:
            print("User already existed and was signed in")
            # TODO: Post sign in logic
        
        return result

    original_implementation.sign_in_up = thirdparty_sign_in_up

    return original_implementation

# this is the location of the SuperTokens core.
supertokens_config = SupertokensConfig(
    connection_uri="https://st-dev-1ae723d0-2666-11ef-ad47-516b0aeb722e.aws.supertokens.io",
    api_key='hhZ5je2pgGeKy0yOoEJHTcjSeV')


app_info = InputAppInfo(
    app_name="Supertokens",
    api_domain="http://localhost:3001",
    website_domain="http://localhost:3000",
)

framework = "fastapi"

# recipeList contains all the modules that you want to
# use from SuperTokens. See the full list here: https://supertokens.com/docs/guides
recipe_list = [
    session.init(),
    emailpassword.init(
        override=emailpassword.InputOverrideConfig(
                functions=override_emailpassword_functions
            ),
    ),
    thirdparty.init(
        override=thirdparty.InputOverrideConfig(
                functions=override_thirdparty_functions
            ),
        sign_in_and_up_feature=thirdparty.SignInAndUpFeature(
            providers=[
            ProviderInput(
                config=ProviderConfig(
                    third_party_id="google",
                    clients=[
                        ProviderClientConfig(
                            client_id='1060725074195-kmeum4crr01uirfl2op9kd5acmi9jutn.apps.googleusercontent.com',
                            client_secret='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'
                        ),
                    ],
                ),
            ),
            ProviderInput(
                config=ProviderConfig(
                    third_party_id="github",
                    clients=[
                        ProviderClientConfig(
                            client_id='467101b197249757c71f',
                            client_secret='e97051221f4b6426e8fe8d51486396703012f5bd'
                        ),
                    ],
                ),
            ),
        ])
    ),
    dashboard.init(),
    userroles.init()
]


